from pathlib import Path
import argparse
import csv

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

    # --------------------------------------------------------
    # Frozen MedCLIP tokens
    # --------------------------------------------------------

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

    # Original MedCLIP representation x
    v0 = h_v.mean(dim=1)
    t0 = h_t.mean(dim=1)

    # gamma = 0 means pure MedCLIP
    if gamma == 0.0:
        return v0, t0

    # CDA representation
    v_cda, t_cda = cda_forward_medclip(
        cda,
        h_v,
        h_t,
        attention_mask,
    )

    # --------------------------------------------------------
    # Teacher's new interpolation structure:
    #
    # output = x + gamma*(CDA(x)-x)
    #        = (1-gamma)*x + gamma*CDA(x)
    # --------------------------------------------------------

    v = (
        (1.0 - gamma) * v0
        + gamma * v_cda
    )

    t = (
        (1.0 - gamma) * t0
        + gamma * t_cda
    )

    return v, t


def contrastive_loss(v, t):

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


@torch.no_grad()
def evaluate(
    loader,
    medclip,
    processor,
    cda,
    device,
    gamma,
):

    cda.eval()

    all_v = []
    all_t = []

    for paths, texts in loader:

        v, t = forward_batch(
            paths,
            texts,
            medclip,
            processor,
            cda,
            device,
            gamma,
        )

        all_v.append(
            F.normalize(
                v.float(),
                dim=-1,
            ).cpu()
        )

        all_t.append(
            F.normalize(
                t.float(),
                dim=-1,
            ).cpu()
        )

    v = torch.cat(
        all_v,
        dim=0,
    )

    t = torch.cat(
        all_t,
        dim=0,
    )

    logits = (
        v @ t.T
    ) / TAU

    gt = torch.arange(
        len(v)
    )

    val_loss = 0.5 * (
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

    i2t_pred = logits.argmax(
        dim=1
    )

    t2i_pred = logits.T.argmax(
        dim=1
    )

    i2t_r1 = (
        i2t_pred == gt
    ).float().mean().item()

    t2i_r1 = (
        t2i_pred == gt
    ).float().mean().item()

    bi_r1 = (
        i2t_r1 + t2i_r1
    ) / 2.0

    return (
        val_loss.item(),
        i2t_r1,
        t2i_r1,
        bi_r1,
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--gamma",
        type=float,
        required=True,
    )

    p.add_argument(
        "--fold",
        type=int,
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

    seed_all(
        SEED
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required"
        )

    device = torch.device(
        "cuda"
    )

    train_file = (
        ROOT
        / f"data/MIMIC_CXR_GAZE_fold{args.fold}_train.txt"
    )

    val_file = (
        ROOT
        / f"data/MIMIC_CXR_GAZE_fold{args.fold}_val.txt"
    )

    print("=" * 72)
    print("CDA-MedCLIP GAMMA CV")
    print("=" * 72)
    print("gamma:", args.gamma)
    print("fold:", args.fold)
    print("train:", train_file)
    print("val:", val_file)
    print("LR:", args.lr)
    print("WD:", args.wd)
    print("Batch:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("MedCLIP frozen: YES")
    print(
        "output = (1-gamma)*x + gamma*CDA(x)"
    )

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

    medclip.load_state_dict(
        state
    )

    medclip = medclip.to(
        device
    )

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

    # ========================================================
    # Data
    # ========================================================

    train_ds = PairDataset(
        train_file
    )

    val_ds = PairDataset(
        val_file
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED + args.fold
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=4,
        collate_fn=collate,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
        drop_last=False,
    )

    print(
        "Train pairs:",
        len(train_ds),
    )

    print(
        "Val pairs:",
        len(val_ds),
    )

    gamma_tag = (
        str(args.gamma)
        .replace(".", "p")
    )

    ckpt = (
        ROOT
        / "checkpoints/cda_gamma_cv"
        / f"gamma_{gamma_tag}_fold{args.fold}.pt"
    )

    result_file = (
        ROOT
        / "results/cda_gamma_cv"
        / f"gamma_{gamma_tag}_fold{args.fold}.csv"
    )

    # ========================================================
    # gamma = 0:
    # pure MedCLIP; no CDA training needed
    # ========================================================

    if args.gamma == 0.0:

        val_loss, i2t, t2i, bi = evaluate(
            val_loader,
            medclip,
            processor,
            cda,
            device,
            args.gamma,
        )

        best_epoch = 0
        best_val = val_loss
        best_i2t = i2t
        best_t2i = t2i
        best_bi = bi

        print()
        print(
            "gamma=0 -> pure MedCLIP, "
            "CDA training skipped"
        )

    else:

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

        best_val = float("inf")
        best_epoch = -1
        best_i2t = 0.0
        best_t2i = 0.0
        best_bi = 0.0

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            cda.train()

            total = 0.0
            n = 0

            for paths, texts in train_loader:

                v, t = forward_batch(
                    paths,
                    texts,
                    medclip,
                    processor,
                    cda,
                    device,
                    args.gamma,
                )

                loss = contrastive_loss(
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

            (
                val_loss,
                i2t,
                t2i,
                bi,
            ) = evaluate(
                val_loader,
                medclip,
                processor,
                cda,
                device,
                args.gamma,
            )

            print(
                f"epoch {epoch:02d} "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f} "
                f"I2T_R1={i2t:.4f} "
                f"T2I_R1={t2i:.4f} "
                f"BI_R1={bi:.4f}"
            )

            if val_loss < best_val:

                best_val = val_loss
                best_epoch = epoch
                best_i2t = i2t
                best_t2i = t2i
                best_bi = bi

                torch.save(
                    cda.state_dict(),
                    ckpt,
                )

                print(
                    "[BEST]",
                    "epoch=",
                    best_epoch,
                    "val_loss=",
                    f"{best_val:.4f}",
                )

    # ========================================================
    # Save fold result
    # ========================================================

    row = {
        "gamma": args.gamma,
        "fold": args.fold,
        "lr": args.lr,
        "wd": args.wd,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "i2t_r1": best_i2t,
        "t2i_r1": best_t2i,
        "bidirectional_r1": best_bi,
    }

    with result_file.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        writer.writeheader()
        writer.writerow(row)

    print()
    print("=" * 72)
    print("FINAL FOLD RESULT")
    print("=" * 72)

    for k, v in row.items():
        print(
            f"{k}: {v}"
        )

    print(
        "Saved:",
        result_file,
    )


if __name__ == "__main__":
    main()
