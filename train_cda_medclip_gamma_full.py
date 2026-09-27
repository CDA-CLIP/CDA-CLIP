from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from medclip import (
    MedCLIPModel,
    MedCLIPVisionModel,
    MedCLIPProcessor,
)

from CDA import CDA

from train_cda_medclip import (
    PairDataset,
    collate,
    get_image_tokens,
    get_text_tokens,
    cda_forward_medclip,
    seed_all,
)


ROOT = Path(".")

TRAIN_FILE = ROOT / "data/MIMIC_CXR_GAZE_SUBSET_753.txt"

MEDCLIP_CKPT = (
    ROOT
    / "pretrained/medclip-resnet/pytorch_model.bin"
)

BATCH_SIZE = 16
EPOCHS = 50
SEED = 42
TAU = 0.07


def load_images(paths):

    images = []

    for p in paths:

        with Image.open(p) as im:

            images.append(
                im.convert("RGB").copy()
            )

    return images


def forward_batch(
    paths,
    texts,
    medclip,
    processor,
    cda,
    device,
    gamma,
):

    images = load_images(paths)

    img_inp = processor(
        images=images,
        return_tensors="pt",
    )

    txt_inp = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    pixels = img_inp[
        "pixel_values"
    ].to(device)

    input_ids = txt_inp[
        "input_ids"
    ].to(device)

    attention_mask = txt_inp[
        "attention_mask"
    ].to(device)

    # Frozen MedCLIP
    with torch.no_grad():

        h_v = get_image_tokens(
            medclip,
            pixels,
        ).float()

        h_t = get_text_tokens(
            medclip,
            input_ids,
            attention_mask,
        ).float()

    # original MedCLIP representation x
    v0 = h_v.mean(dim=1)
    t0 = h_t.mean(dim=1)

    if gamma == 0.0:

        return v0, t0

    # CDA(x)
    v_cda, t_cda = cda_forward_medclip(
        cda,
        h_v,
        h_t,
        attention_mask,
    )

    # output = (1-gamma)x + gamma*CDA(x)

    v = (
        (1.0 - gamma) * v0
        + gamma * v_cda
    )

    t = (
        (1.0 - gamma) * t0
        + gamma * t_cda
    )

    return v, t


def loss_fn(v, t):

    v = F.normalize(
        v.float(),
        dim=-1,
    )

    t = F.normalize(
        t.float(),
        dim=-1,
    )

    logits = (
        v @ t.T
    ) / TAU

    gt = torch.arange(
        len(v),
        device=v.device,
    )

    return 0.5 * (
        F.cross_entropy(
            logits,
            gt,
        )
        +
        F.cross_entropy(
            logits.T,
            gt,
        )
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--gamma",
        type=float,
        required=True,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--wd",
        type=float,
        default=0.01,
    )

    args = p.parse_args()

    seed_all(SEED)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "GPU required"
        )

    device = torch.device("cuda")

    gamma_tag = (
        str(args.gamma)
        .replace(".", "p")
    )

    print("=" * 72)
    print("FULL 753 GAMMA TRAINING")
    print("=" * 72)
    print("gamma:", args.gamma)
    print(
        "output = (1-gamma)x + gamma*CDA(x)"
    )
    print("data:", TRAIN_FILE)
    print("LR:", args.lr)
    print("WD:", args.wd)
    print("Batch:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("MedCLIP frozen: YES")

    # ========================================================
    # Frozen MedCLIP
    # ========================================================

    medclip = MedCLIPModel(
        vision_cls=MedCLIPVisionModel
    )

    state = torch.load(
        MEDCLIP_CKPT,
        map_location="cpu",
    )

    medclip.load_state_dict(state)

    medclip = medclip.to(device)

    medclip.eval()

    for p in medclip.parameters():

        p.requires_grad = False

    processor = MedCLIPProcessor()

    # ========================================================
    # CDA
    # ========================================================

    cda = CDA(
        d=512,
        h=8,
        d_k=64,
        num_layers=1,
        pool="mean",
        use_proj_heads=True,
    ).to(device)

    ckpt = (
        ROOT
        / "checkpoints/cda_gamma_full"
        / f"gamma_{gamma_tag}.pt"
    )

    # gamma=0 = pure MedCLIP
    # CDA is irrelevant, just save state so evaluator can load it
    if args.gamma == 0.0:

        torch.save(
            cda.state_dict(),
            ckpt,
        )

        print(
            "gamma=0: CDA training skipped"
        )

        print(
            "Checkpoint:",
            ckpt,
        )

        return

    optimizer = torch.optim.AdamW(
        cda.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=1e-7,
        )
    )

    # ========================================================
    # Full 753
    # ========================================================

    ds = PairDataset(
        TRAIN_FILE
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=4,
        collate_fn=collate,
        drop_last=False,
    )

    print(
        "Training pairs:",
        len(ds),
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        cda.train()

        total = 0.0
        n = 0

        for paths, texts in loader:

            v, t = forward_batch(
                paths,
                texts,
                medclip,
                processor,
                cda,
                device,
                args.gamma,
            )

            loss = loss_fn(
                v,
                t,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            bs = len(paths)

            total += (
                loss.item() * bs
            )

            n += bs

        scheduler.step()

        train_loss = total / n

        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.6f}"
        )

    torch.save(
        cda.state_dict(),
        ckpt,
    )

    print()
    print("TRAINING COMPLETE")
    print(
        "Checkpoint:",
        ckpt,
    )


if __name__ == "__main__":
    main()
