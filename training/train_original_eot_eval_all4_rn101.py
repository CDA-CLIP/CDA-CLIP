
# ---------------------------------------------------------------------------
# Repository-local import bootstrap
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _local_path in (
    _REPO_ROOT,
    _REPO_ROOT / "training",
    _REPO_ROOT / "evaluation",
    _REPO_ROOT / "analysis",
):
    _local_path = str(_local_path)
    if _local_path not in _sys.path:
        _sys.path.insert(0, _local_path)

import sys
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from CDA import CDA
from eval_pairwise_eot_all4 import (
    load_chest,
    load_siim,
    load_inbreast,
    load_chex,
    evaluate_dataset,
)

TRAIN_TXT = ROOT / "data/MIMIC_CXR_GAZE_SUBSET_753.txt"
MIMIC_ROOT = ROOT / "data/MIMIC_CXR/raw"

CKPT = (
    ROOT
    / "checkpoints"
    / "cda_original_eot753_rn101_lr2e4_wd5_bs16_seed42.pt"
)

OUT_CSV = (
    ROOT
    / "results"
    / "original_eot_rn101_all4_results.csv"
)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 5.0
TAU = 0.07


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SOURCE DATASET
# ============================================================

class SourceDataset(Dataset):

    def __init__(self, txt_file, preprocess):
        self.preprocess = preprocess
        self.samples = []

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                if not line:
                    continue

                parts = line.split("^", 1)

                if len(parts) != 2:
                    continue

                path, report = parts
                self.samples.append((path, report))

        print("Source samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def resolve_path(self, raw_path):
        p = Path(raw_path)

        if p.exists():
            return p

        normalized = raw_path.replace("\\", "/")

        marker = "/MIMIC_CXR/raw/"

        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            candidate = MIMIC_ROOT / rel

            if candidate.exists():
                return candidate

        # fallback only if needed
        matches = list(MIMIC_ROOT.rglob(p.name))

        if len(matches) > 0:
            return matches[0]

        raise FileNotFoundError(raw_path)

    def __getitem__(self, idx):
        raw_path, report = self.samples[idx]

        path = self.resolve_path(raw_path)

        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)

        return image, report


# ============================================================
# FROZEN CLIP TOKEN ENCODERS
# ============================================================

@torch.no_grad()
def encode_image_tokens(model, image: torch.Tensor) -> torch.Tensor:
    """
    Return frozen CLIP visual tokens in 512-d shared space.

    ViT:
        patch/CLS tokens -> visual.proj -> 512 d

    ModifiedResNet:
        layer4 spatial tokens + global token ->
        frozen CLIP attention-pool projections -> 512 d
    """
    visual = model.visual

    # =========================
    # ViT branch
    # =========================
    if hasattr(visual, "transformer"):
        x = visual.conv1(image.type(model.dtype))
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        ).permute(0, 2, 1)

        cls = (
            visual.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        )

        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = visual.ln_post(x)

        if visual.proj is not None:
            x = x @ visual.proj

    # =========================
    # RN50 / RN101 branch
    # =========================
    else:
        def stem(z):
            z = visual.relu1(visual.bn1(visual.conv1(z)))
            z = visual.relu2(visual.bn2(visual.conv2(z)))
            z = visual.relu3(visual.bn3(visual.conv3(z)))
            z = visual.avgpool(z)
            return z

        x = image.type(model.dtype)

        x = stem(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)

        # B,C,H,W -> B,HW,C
        x = x.flatten(2).permute(0, 2, 1)

        # Global token + spatial tokens
        cls = x.mean(dim=1, keepdim=True)
        x = torch.cat([cls, x], dim=1)

        pos = visual.attnpool.positional_embedding.to(
            dtype=x.dtype,
            device=x.device,
        )

        if x.shape[1] != pos.shape[0]:
            raise RuntimeError(
                f"RN token/position mismatch: "
                f"tokens={tuple(x.shape)}, "
                f"pos={tuple(pos.shape)}"
            )

        x = x + pos.unsqueeze(0)

        # RN101 layer4 tokens are 2048-d.
        # Reuse frozen CLIP attention-pool projections,
        # instead of introducing a newly trained projection.
        x = visual.attnpool.v_proj(x)
        x = visual.attnpool.c_proj(x)

    if x.ndim != 3:
        raise RuntimeError(
            f"Visual token tensor must be 3-D, got {tuple(x.shape)}"
        )

    if x.shape[-1] != 512:
        raise RuntimeError(
            f"Visual tokens must be 512-d before CDA, "
            f"got {tuple(x.shape)}"
        )

    return x


@torch.no_grad()
def encode_text_tokens(model, text_tokens):
    x = model.token_embedding(
        text_tokens
    ).type(model.dtype)

    x = (
        x
        + model.positional_embedding
        .type(model.dtype)
    )

    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)

    x = model.ln_final(
        x
    ).type(model.dtype)

    x = x @ model.text_projection

    return x.float()


# ============================================================
# CDA WITH CORRECT CLIP TEXT EOT POOLING
# ============================================================

def cda_forward_eot(
    cda,
    h_v,
    h_t,
    text_tokens
):

    # RN101_DTYPE_DIM_GUARD
    h_v = h_v.float()
    h_t = h_t.float()
    if h_v.shape[-1] != 512 or h_t.shape[-1] != 512:
        raise RuntimeError(
            f'CDA expects 512-d tokens: visual={tuple(h_v.shape)}, text={tuple(h_t.shape)}'
        )
    for layer in cda.layers:
        h_v, h_t = layer(
            h_v,
            h_t
        )

    h_v = cda.norm_v(h_v)
    h_t = cda.norm_t(h_t)

    # ViT image global token = CLS
    v_g = h_v[:, 0]

    # CLIP text global token = EOT
    eot_idx = text_tokens.argmax(
        dim=-1
    )

    batch_idx = torch.arange(
        h_t.shape[0],
        device=h_t.device
    )

    t_g = h_t[
        batch_idx,
        eot_idx
    ]

    if cda.W_v is not None:
        v_g = cda.W_v(v_g)
        t_g = cda.W_t(t_g)

    return v_g, t_g


# ============================================================
# ORIGINAL CDA TRAINING OBJECTIVE
# ============================================================

def train():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    clip_model, preprocess = clip.load(
        "RN101",
        device=device,
        jit=False
    )

    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    cda = CDA(
        d=512,
        h=8,
        d_k=64,
        num_layers=1,
        pool="cls",
        use_proj_heads=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        cda.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    dataset = SourceDataset(
        TRAIN_TXT,
        preprocess
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
    )

    print()
    print("=" * 80)
    print("TRAINING: ORIGINAL OBJECTIVE + EOT")
    print("=" * 80)

    for epoch in range(EPOCHS):

        cda.train()

        total_loss = 0.0
        total_n = 0

        for images, reports in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            text_tokens = clip.tokenize(
                list(reports),
                truncate=True
            ).to(device)

            # Frozen CLIP encoders
            h_v = encode_image_tokens(
                clip_model,
                images
            )

            h_t = encode_text_tokens(
                clip_model,
                text_tokens
            )

            # IMPORTANT:
            # Original training behavior:
            # CDA is run only on matched
            # Image_i / Report_i pairs.
            v_g, t_g = cda_forward_eot(
                cda,
                h_v,
                h_t,
                text_tokens
            )

            v_g = F.normalize(
                v_g,
                dim=-1
            )

            t_g = F.normalize(
                t_g,
                dim=-1
            )

            # Original CLIP-style in-batch
            # contrastive similarity matrix
            logits = (
                v_g @ t_g.t()
            ) / TAU

            labels = torch.arange(
                logits.shape[0],
                device=device
            )

            loss_i = F.cross_entropy(
                logits,
                labels
            )

            loss_t = F.cross_entropy(
                logits.t(),
                labels
            )

            loss = (
                loss_i + loss_t
            ) / 2.0

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            bs = images.shape[0]

            total_loss += (
                loss.item() * bs
            )

            total_n += bs

        epoch_loss = (
            total_loss / total_n
        )

        print(
            f"epoch {epoch+1}/{EPOCHS} "
            f"loss={epoch_loss:.6f}"
        )

    CKPT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "cda_state_dict":
                cda.state_dict(),

            "vision_encoder":
                "RN101",

            "num_cda_layers":
                1,

            "num_heads":
                8,

            "lr":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "seed":
                SEED,

            "tau":
                TAU,

            "text_pooling":
                "EOT",

            "training_objective":
                "original_matched_pair_CDA_then_inbatch_contrastive",
        },
        CKPT
    )

    print()
    print(
        "Saved checkpoint:",
        CKPT
    )

    return (
        clip_model,
        cda,
        preprocess,
        device
    )


# ============================================================
# FOUR-DATASET EVALUATION
# ============================================================

def evaluate_all(
    clip_model,
    cda,
    preprocess,
    device
):

    cda.eval()

    datasets = [
        ("ChestX-ray",)
        + load_chest(),

        ("SIIM-ACR",)
        + load_siim(),

        ("INbreast",)
        + load_inbreast(),

        ("CheXpert5x200",)
        + load_chex(),
    ]

    results = []

    print()
    print("=" * 80)
    print(
        "EVALUATION: "
        "ORIGINAL OBJECTIVE + EOT"
    )
    print("=" * 80)

    for (
        name,
        classes,
        samples
    ) in datasets:

        result = evaluate_dataset(
            name,
            classes,
            samples,
            clip_model,
            cda,
            preprocess,
            device
        )

        results.append(result)

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUT_CSV.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print(
        "FINAL TABLE — "
        "ORIGINAL OBJECTIVE + EOT"
    )
    print("=" * 80)

    print(
        f"{'Dataset':<18}"
        f"{'N':>7}"
        f"{'Single BA':>14}"
        f"{'7P BA':>12}"
    )

    for r in results:
        print(
            f"{r['dataset']:<18}"
            f"{r['n']:>7}"
            f"{r['single_ba']:>13.2f}%"
            f"{r['ensemble_ba']:>11.2f}%"
        )

    print()
    print(
        "Saved results:",
        OUT_CSV
    )


def main():

    (
        clip_model,
        cda,
        preprocess,
        device
    ) = train()

    evaluate_all(
        clip_model,
        cda,
        preprocess,
        device
    )


if __name__ == "__main__":
    main()
