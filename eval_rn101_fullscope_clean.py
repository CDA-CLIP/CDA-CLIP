import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from CDA import CDA

# ------------------------------------------------------------
# IMPORTANT:
# We use the VERIFIED full-scope loaders/templates only.
# We DO NOT use its get_scores/evaluate_dataset.
# ------------------------------------------------------------
sys.path.insert(0, str(ROOT / "baseline_suite"))

from fullscope_data import (
    load_chest,
    load_siim,
    load_inbreast,
    load_chexpert,
)


CKPT = (
    ROOT
    / "checkpoints"
    / "cda_original_eot753_rn101_lr2e4_wd5_bs16_seed42.pt"
)

OUT_CSV = (
    ROOT
    / "results"
    / "original_eot_rn101_fullscope_results.csv"
)

TEMPLATES = [
    "This is an image of {c}.",
    "A chest X-ray showing {c}.",
    "Radiograph with findings consistent with {c}.",
    "A medical scan demonstrating {c}.",
    "Imaging study of a patient with {c}.",
    "An X-ray of a patient diagnosed with {c}.",
    "{c} visible on this image.",
]

EXPECTED_N = {
    "ChestXray": 5856,
    "SIIM": 1250,
    "INbreast": 6154,
    "CheXpert5x200": 1000,
}


# ============================================================
# DATASET
# ============================================================

class EvalDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)

        return image, label


# ============================================================
# RN101 IMAGE TOKENS
# EXACT SAME IMPLEMENTATION USED TO TRAIN THIS CHECKPOINT
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


# ============================================================
# TEXT TOKENS
# EXACT SAME EOT TRAINING IMPLEMENTATION
# ============================================================

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
# CDA FORWARD
# EXACT SAME IMPLEMENTATION USED DURING RN101 TRAINING
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
# PAIR-CONDITIONED SINGLE / 7-PROMPT SCORING
# ============================================================

@torch.no_grad()
def get_scores(
    images,
    classes,
    templates,
    clip_model,
    cda,
    device,
):

    images = images.to(device)

    h_v = encode_image_tokens(
        clip_model,
        images
    )

    all_class_scores = []

    for class_name in classes:

        v_embeddings = []
        t_embeddings = []

        for template in templates:

            prompt = template.format(
                c=class_name
            )

            tokens = clip.tokenize(
                [prompt] * len(images),
                truncate=True
            ).to(device)

            h_t = encode_text_tokens(
                clip_model,
                tokens
            )

            # Pair-conditioned CDA:
            # visual representation depends on candidate text.
            v_g, t_g = cda_forward_eot(
                cda,
                h_v,
                h_t,
                tokens
            )

            v_embeddings.append(v_g)
            t_embeddings.append(t_g)

        v_g = torch.stack(
            v_embeddings,
            dim=0
        ).mean(dim=0)

        t_g = torch.stack(
            t_embeddings,
            dim=0
        ).mean(dim=0)

        v_g = F.normalize(
            v_g,
            dim=-1
        )

        t_g = F.normalize(
            t_g,
            dim=-1
        )

        score = (
            v_g * t_g
        ).sum(dim=-1)

        all_class_scores.append(
            score
        )

    return torch.stack(
        all_class_scores,
        dim=1
    )


# ============================================================
# METRICS
# ============================================================

def summarize(
    y_true,
    scores,
    n_classes,
):

    pred = scores.argmax(dim=1)

    recalls = []

    for i in range(n_classes):

        mask = y_true == i

        if mask.sum() == 0:
            recalls.append(
                float("nan")
            )
        else:
            recalls.append(
                float(
                    (pred[mask] == i)
                    .float()
                    .mean()
                )
            )

    ba = float(
        np.nanmean(recalls)
    )

    acc = float(
        (pred == y_true)
        .float()
        .mean()
    )

    counts = torch.bincount(
        pred,
        minlength=n_classes
    ).tolist()

    return (
        ba,
        acc,
        recalls,
        counts,
    )


# ============================================================
# DATASET EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    name,
    classes,
    samples,
    clip_model,
    cda,
    preprocess,
    device,
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("N =", len(samples))

    if len(samples) != EXPECTED_N[name]:
        raise RuntimeError(
            f"{name}: expected "
            f"{EXPECTED_N[name]}, "
            f"got {len(samples)}"
        )

    true_counts = Counter(
        y for _, y in samples
    )

    print(
        "True counts:",
        {
            classes[i]: true_counts[i]
            for i in range(len(classes))
        }
    )

    ds = EvalDataset(
        samples,
        preprocess
    )

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    y_all = []
    single_all = []
    ensemble_all = []

    for batch_i, (images, labels) in enumerate(loader):

        single = get_scores(
            images,
            classes,
            [TEMPLATES[0]],
            clip_model,
            cda,
            device,
        )

        ensemble = get_scores(
            images,
            classes,
            TEMPLATES,
            clip_model,
            cda,
            device,
        )

        if not torch.isfinite(single).all():
            raise RuntimeError(
                f"{name}: non-finite Single scores"
            )

        if not torch.isfinite(ensemble).all():
            raise RuntimeError(
                f"{name}: non-finite 7P scores"
            )

        y_all.append(
            labels.cpu()
        )

        single_all.append(
            single.cpu()
        )

        ensemble_all.append(
            ensemble.cpu()
        )

        if batch_i == 0:
            print(
                "First batch passed:",
                "single",
                tuple(single.shape),
                "| 7P",
                tuple(ensemble.shape),
            )

    y_true = torch.cat(
        y_all
    )

    single_scores = torch.cat(
        single_all
    )

    ensemble_scores = torch.cat(
        ensemble_all
    )

    (
        s_ba,
        s_acc,
        s_rec,
        s_counts,
    ) = summarize(
        y_true,
        single_scores,
        len(classes),
    )

    (
        e_ba,
        e_acc,
        e_rec,
        e_counts,
    ) = summarize(
        y_true,
        ensemble_scores,
        len(classes),
    )

    print(
        f"Single BA={100*s_ba:.2f}%  "
        f"ACC={100*s_acc:.2f}%"
    )

    print(
        "Single predicted:",
        dict(
            zip(
                classes,
                s_counts
            )
        ),
    )

    print(
        f"7P BA={100*e_ba:.2f}%  "
        f"ACC={100*e_acc:.2f}%"
    )

    print(
        "7P predicted:",
        dict(
            zip(
                classes,
                e_counts
            )
        ),
    )

    return {
        "dataset": name,
        "n": len(samples),
        "single_ba": 100 * s_ba,
        "ensemble_ba": 100 * e_ba,
        "single_acc": 100 * s_acc,
        "ensemble_acc": 100 * e_acc,
        "single_pred_counts":
            str(dict(zip(classes, s_counts))),
        "ensemble_pred_counts":
            str(dict(zip(classes, e_counts))),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CDA-CLIP RN101")
    print("EXTERNAL TEST ONLY")
    print("NO TRAINING")
    print("NO ARCHIVE SCORING FUNCTIONS")
    print("=" * 80)

    if not CKPT.exists():
        raise FileNotFoundError(
            CKPT
        )

    if len(TEMPLATES) != 7:
        raise RuntimeError(
            f"Expected 7 prompts, "
            f"found {len(TEMPLATES)}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", CKPT)
    print("Number of prompts:", len(TEMPLATES))

    clip_model, preprocess = clip.load(
        "RN101",
        device=device,
        jit=False,
    )

    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    obj = torch.load(
        CKPT,
        map_location="cpu",
    )

    print(
        "Checkpoint keys:",
        list(obj.keys())
    )

    print(
        "Checkpoint vision_encoder:",
        obj.get("vision_encoder")
    )

    cda = CDA(
        d=512,
        h=8,
        d_k=64,
        num_layers=1,
        pool="cls",
        use_proj_heads=True,
    ).to(device)

    cda.load_state_dict(
        obj["cda_state_dict"],
        strict=True,
    )

    cda.eval()

    # --------------------------------------------------------
    # Load the already-verified FULL-SCOPE datasets.
    # --------------------------------------------------------

    datasets = [
        load_chest(),
        load_siim(),
        load_inbreast(),
        load_chexpert(),
    ]

    print()
    print("FULL-SCOPE COUNTS")

    for name, classes, samples in datasets:

        print(
            f"{name}: "
            f"{len(samples)} "
            f"(expected {EXPECTED_N[name]})"
        )

        if len(samples) != EXPECTED_N[name]:
            raise RuntimeError(
                f"{name}: wrong N"
            )

    # --------------------------------------------------------
    # GPU SMOKE TEST BEFORE FULL EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SMOKE TEST — 2 CHEST IMAGES")
    print("=" * 80)

    (
        smoke_name,
        smoke_classes,
        smoke_samples,
    ) = datasets[0]

    smoke_ds = EvalDataset(
        smoke_samples[:2],
        preprocess,
    )

    smoke_images = torch.stack(
        [
            smoke_ds[0][0],
            smoke_ds[1][0],
        ],
        dim=0,
    )

    smoke_single = get_scores(
        smoke_images,
        smoke_classes,
        [TEMPLATES[0]],
        clip_model,
        cda,
        device,
    )

    smoke_7p = get_scores(
        smoke_images,
        smoke_classes,
        TEMPLATES,
        clip_model,
        cda,
        device,
    )

    expected_shape = (
        2,
        len(smoke_classes),
    )

    if tuple(smoke_single.shape) != expected_shape:
        raise RuntimeError(
            f"Smoke Single shape wrong: "
            f"{tuple(smoke_single.shape)}"
        )

    if tuple(smoke_7p.shape) != expected_shape:
        raise RuntimeError(
            f"Smoke 7P shape wrong: "
            f"{tuple(smoke_7p.shape)}"
        )

    if not torch.isfinite(
        smoke_single
    ).all():
        raise RuntimeError(
            "Smoke Single contains non-finite values"
        )

    if not torch.isfinite(
        smoke_7p
    ).all():
        raise RuntimeError(
            "Smoke 7P contains non-finite values"
        )

    print(
        "SMOKE TEST = PASS"
    )

    print(
        "Single shape:",
        tuple(smoke_single.shape)
    )

    print(
        "7P shape:",
        tuple(smoke_7p.shape)
    )

    # --------------------------------------------------------
    # FULL EXTERNAL EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING FULL EXTERNAL EVALUATION")
    print("=" * 80)

    results = []

    for (
        name,
        classes,
        samples,
    ) in datasets:

        results.append(
            evaluate_dataset(
                name,
                classes,
                samples,
                clip_model,
                cda,
                preprocess,
                device,
            )
        )

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print("FINAL RN101 FULL-SCOPE RESULTS")
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
        "Saved:",
        OUT_CSV
    )


if __name__ == "__main__":
    main()
