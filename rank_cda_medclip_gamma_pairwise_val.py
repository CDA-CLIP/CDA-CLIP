from pathlib import Path
import csv
import statistics

import torch
import torch.nn.functional as F
from PIL import Image

from medclip import (
    MedCLIPModel,
    MedCLIPVisionModel,
    MedCLIPProcessor,
)

from CDA import CDA

from train_cda_medclip import (
    get_image_tokens,
    get_text_tokens,
    cda_forward_medclip,
)


ROOT = Path(".")

MEDCLIP_CKPT = (
    ROOT
    / "pretrained/medclip-resnet/pytorch_model.bin"
)

CKPT_DIR = (
    ROOT
    / "checkpoints/cda_gamma_cv"
)

OUT_DIR = (
    ROOT
    / "results/cda_gamma_pairwise"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GAMMAS = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
]

FOLDS = [1, 2, 3]

TAU = 0.07
IMAGE_BATCH = 32
PAIR_BATCH = 64


def gamma_tag(gamma):
    return str(gamma).replace(".", "p")


def read_pairs(path):

    rows = []

    with open(path) as f:
        for line in f:

            line = line.rstrip("\n")

            if not line:
                continue

            image_path, report = line.split(
                "^",
                1,
            )

            rows.append(
                (
                    image_path,
                    report,
                )
            )

    return rows


def load_images(paths):

    images = []

    for p in paths:

        with Image.open(p) as im:

            images.append(
                im.convert("RGB").copy()
            )

    return images


@torch.no_grad()
def precompute_fold(
    val_rows,
    medclip,
    processor,
    device,
):

    image_paths = [
        r[0]
        for r in val_rows
    ]

    reports = [
        r[1]
        for r in val_rows
    ]

    # ========================================================
    # IMAGE TOKENS
    # ========================================================

    all_image_tokens = []

    for start in range(
        0,
        len(image_paths),
        IMAGE_BATCH,
    ):

        paths = image_paths[
            start:start + IMAGE_BATCH
        ]

        images = load_images(
            paths
        )

        inp = processor(
            images=images,
            return_tensors="pt",
        )

        pixels = inp[
            "pixel_values"
        ].to(device)

        h_v = get_image_tokens(
            medclip,
            pixels,
        ).float()

        all_image_tokens.append(
            h_v
        )

    image_tokens = torch.cat(
        all_image_tokens,
        dim=0,
    )

    # ========================================================
    # TEXT TOKENS
    # each candidate separately
    # ========================================================

    text_tokens = []
    text_masks = []

    for report in reports:

        inp = processor(
            text=[report],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inp[
            "input_ids"
        ].to(device)

        mask = inp[
            "attention_mask"
        ].to(device)

        h_t = get_text_tokens(
            medclip,
            input_ids,
            mask,
        ).float()

        text_tokens.append(
            h_t
        )

        text_masks.append(
            mask
        )

    return (
        image_tokens,
        text_tokens,
        text_masks,
    )


@torch.no_grad()
def evaluate_gamma(
    gamma,
    fold,
    val_rows,
    image_tokens,
    text_tokens,
    text_masks,
    device,
):

    n = len(val_rows)

    # ========================================================
    # gamma = 0
    # pure MedCLIP pathway, no CDA checkpoint
    # ========================================================

    if gamma == 0.0:

        v0 = image_tokens.mean(
            dim=1
        )

        t0 = torch.cat(
            [
                x.mean(
                    dim=1
                )
                for x in text_tokens
            ],
            dim=0,
        )

        v0 = F.normalize(
            v0.float(),
            dim=-1,
        )

        t0 = F.normalize(
            t0.float(),
            dim=-1,
        )

        score_matrix = (
            v0 @ t0.T
        )

    # ========================================================
    # gamma > 0
    # TRUE PAIRWISE CDA
    # ========================================================

    else:

        tag = gamma_tag(
            gamma
        )

        ckpt = (
            CKPT_DIR
            / f"gamma_{tag}_fold{fold}.pt"
        )

        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {ckpt}"
            )

        cda = CDA(
            d=512,
            h=8,
            d_k=64,
            num_layers=1,
            pool="mean",
            use_proj_heads=True,
        ).to(device)

        cda.load_state_dict(
            torch.load(
                ckpt,
                map_location="cpu",
            )
        )

        cda.eval()

        score_matrix = torch.empty(
            n,
            n,
            device=device,
        )

        # ----------------------------------------------------
        # Column j:
        # every image independently paired
        # with candidate report j
        # ----------------------------------------------------

        for j in range(n):

            if (
                j % 25 == 0
                or j == n - 1
            ):
                print(
                    f"    candidate "
                    f"{j + 1}/{n}"
                )

            h_t_one = text_tokens[j]
            mask_one = text_masks[j]

            for start in range(
                0,
                n,
                PAIR_BATCH,
            ):

                end = min(
                    start + PAIR_BATCH,
                    n,
                )

                h_v = image_tokens[
                    start:end
                ]

                bs = (
                    end - start
                )

                h_t = h_t_one.repeat(
                    bs,
                    1,
                    1,
                )

                mask = mask_one.repeat(
                    bs,
                    1,
                )

                # Original x
                v0 = h_v.mean(
                    dim=1
                )

                t0 = h_t.mean(
                    dim=1
                )

                # CDA(x)
                v_cda, t_cda = (
                    cda_forward_medclip(
                        cda,
                        h_v,
                        h_t,
                        mask,
                    )
                )

                # Teacher interpolation
                # output =
                # (1-gamma)x + gamma*CDA(x)

                v = (
                    (1.0 - gamma) * v0
                    + gamma * v_cda
                )

                t = (
                    (1.0 - gamma) * t0
                    + gamma * t_cda
                )

                v = F.normalize(
                    v.float(),
                    dim=-1,
                )

                t = F.normalize(
                    t.float(),
                    dim=-1,
                )

                score = (
                    v * t
                ).sum(
                    dim=-1
                )

                score_matrix[
                    start:end,
                    j,
                ] = score

        del cda
        torch.cuda.empty_cache()

    # ========================================================
    # TRUE retrieval metrics
    # ========================================================

    labels = torch.arange(
        n,
        device=device,
    )

    i2t_pred = score_matrix.argmax(
        dim=1
    )

    t2i_pred = score_matrix.argmax(
        dim=0
    )

    i2t_r1 = (
        i2t_pred == labels
    ).float().mean().item()

    t2i_r1 = (
        t2i_pred == labels
    ).float().mean().item()

    mean_r1 = (
        i2t_r1
        + t2i_r1
    ) / 2.0

    logits = (
        score_matrix
        / TAU
    )

    loss_i = F.cross_entropy(
        logits,
        labels,
    )

    loss_t = F.cross_entropy(
        logits.T,
        labels,
    )

    pairwise_loss = (
        0.5
        * (
            loss_i
            + loss_t
        )
    ).item()

    return {
        "gamma": gamma,
        "fold": fold,
        "n": n,
        "pairwise_loss": pairwise_loss,
        "i2t_r1": i2t_r1,
        "t2i_r1": t2i_r1,
        "mean_r1": mean_r1,
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required"
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 88)
    print(
        "CDA-MedCLIP GAMMA TRUE PAIRWISE CV"
    )
    print("=" * 88)

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

    all_results = []

    # ========================================================
    # Fold loop
    # ========================================================

    for fold in FOLDS:

        val_file = (
            ROOT
            / f"data/MIMIC_CXR_GAZE_fold{fold}_val.txt"
        )

        val_rows = read_pairs(
            val_file
        )

        print()
        print("#" * 88)
        print(
            f"FOLD {fold} | "
            f"validation pairs = "
            f"{len(val_rows)}"
        )
        print("#" * 88)

        (
            image_tokens,
            text_tokens,
            text_masks,
        ) = precompute_fold(
            val_rows,
            medclip,
            processor,
            device,
        )

        print(
            "Image tokens:",
            tuple(
                image_tokens.shape
            ),
        )

        print(
            "Text candidates:",
            len(text_tokens),
        )

        for gamma in GAMMAS:

            print()
            print(
                "=" * 72
            )

            print(
                f"Fold {fold} "
                f"| gamma={gamma}"
            )

            print(
                "=" * 72
            )

            result = evaluate_gamma(
                gamma,
                fold,
                val_rows,
                image_tokens,
                text_tokens,
                text_masks,
                device,
            )

            all_results.append(
                result
            )

            print(
                f"gamma={gamma} "
                f"fold={fold} "
                f"loss="
                f"{result['pairwise_loss']:.4f} "
                f"I2T_R1="
                f"{result['i2t_r1']:.4f} "
                f"T2I_R1="
                f"{result['t2i_r1']:.4f} "
                f"BI_R1="
                f"{result['mean_r1']:.4f}"
            )

        del image_tokens
        del text_tokens
        del text_masks

        torch.cuda.empty_cache()

    # ========================================================
    # Save by-fold results
    # ========================================================

    by_fold_file = (
        OUT_DIR
        / "gamma_pairwise_by_fold.csv"
    )

    with open(
        by_fold_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma",
                "fold",
                "n",
                "pairwise_loss",
                "i2t_r1",
                "t2i_r1",
                "mean_r1",
            ],
        )

        writer.writeheader()
        writer.writerows(
            all_results
        )

    # ========================================================
    # Aggregate across 3 folds
    # ========================================================

    summary = []

    for gamma in GAMMAS:

        rows = [
            r
            for r in all_results
            if r["gamma"] == gamma
        ]

        losses = [
            r["pairwise_loss"]
            for r in rows
        ]

        i2ts = [
            r["i2t_r1"]
            for r in rows
        ]

        t2is = [
            r["t2i_r1"]
            for r in rows
        ]

        bis = [
            r["mean_r1"]
            for r in rows
        ]

        summary.append(
            {
                "gamma": gamma,
                "mean_pairwise_loss":
                    statistics.mean(losses),
                "sd_pairwise_loss":
                    statistics.stdev(losses),
                "mean_i2t_r1":
                    statistics.mean(i2ts),
                "mean_t2i_r1":
                    statistics.mean(t2is),
                "mean_bi_r1":
                    statistics.mean(bis),
            }
        )

    # PRIMARY SELECTION:
    # lowest true-pairwise source validation loss
    summary.sort(
        key=lambda x:
        x["mean_pairwise_loss"]
    )

    print()
    print("=" * 100)
    print(
        "TRUE PAIRWISE GAMMA CV SUMMARY"
    )
    print(
        "PRIMARY SELECTION = "
        "LOWEST MEAN PAIRWISE VALIDATION LOSS"
    )
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Gamma':<10}"
        f"{'Loss':<12}"
        f"{'SD':<12}"
        f"{'I2T_R1':<12}"
        f"{'T2I_R1':<12}"
        f"{'BI_R1':<12}"
    )

    for rank, r in enumerate(
        summary,
        1,
    ):

        print(
            f"{rank:<6}"
            f"{r['gamma']:<10}"
            f"{r['mean_pairwise_loss']:<12.4f}"
            f"{r['sd_pairwise_loss']:<12.4f}"
            f"{r['mean_i2t_r1']:<12.4f}"
            f"{r['mean_t2i_r1']:<12.4f}"
            f"{r['mean_bi_r1']:<12.4f}"
        )

    selected = summary[0][
        "gamma"
    ]

    print()
    print(
        "SELECTED GAMMA =",
        selected,
    )

    summary_file = (
        OUT_DIR
        / "gamma_pairwise_summary.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma",
                "mean_pairwise_loss",
                "sd_pairwise_loss",
                "mean_i2t_r1",
                "mean_t2i_r1",
                "mean_bi_r1",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary
        )

    print(
        "Saved:",
        by_fold_file,
    )

    print(
        "Saved:",
        summary_file,
    )


if __name__ == "__main__":
    main()
