import sys
import csv
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import clip

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from CDA import CDA
from train_pairwise_cda_eot_eval_inbreast import (
    EvalDataset,
    get_scores,
    TEMPLATES,
)

CKPT = ROOT / "checkpoints/cda_pairwise_eot753_vitb16_lr2e4_wd5_bs16_seed42.pt"

CHEST_ROOT = ROOT / "data/testsets/chest_xray_extracted/chest_xray"
SIIM_ROOT = ROOT / "data/testsets/siim_acr_extracted/SIIM-ACR"
INB_ROOT = ROOT / "data/testsets/inbreast_extracted/INbreast"
CHEX_ROOT = ROOT / "data/testsets/chexpert5x200_extracted/chexpert5x200"

OUT_CSV = ROOT / "results/eot_pairwise_all4_results.csv"


# ============================================================
# DATASETS
# ============================================================

def load_chest():
    classes = ["normal", "pneumonia"]
    samples = []

    for split in ["train", "val", "test"]:
        for label, folder_name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
            folder = CHEST_ROOT / split / folder_name

            if not folder.exists():
                continue

            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples.append((str(p), label))

    return classes, samples


def load_siim():
    # Original released CDA-CLIP mapping:
    # 0 = collapsed lung, 1 = normal lung
    classes = ["collapsed lung", "normal lung"]
    samples = []

    for csv_name in ["train_list.csv", "test_list.csv"]:
        fp = SIIM_ROOT / csv_name

        with fp.open(newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 2:
                    continue

                try:
                    label = int(row[1])
                except ValueError:
                    continue

                img_name = Path(row[0]).name
                p = SIIM_ROOT / img_name

                if p.exists():
                    samples.append((str(p), label))

    return classes, samples


def load_inbreast():
    classes = ["normal", "benign", "malignant"]
    samples = []

    for split in ["train", "test"]:
        folder = INB_ROOT / split

        if not folder.exists():
            continue

        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue

            m = re.search(r"_l([012])(?:_|\.|$)", p.name)

            if m is None:
                continue

            samples.append(
                (str(p), int(m.group(1)))
            )

    return classes, samples


def resolve_chex_path(rel):
    rel = Path(rel)

    candidates = [
        CHEX_ROOT / rel,
    ]

    if len(rel.parts) > 1:
        candidates.append(
            CHEX_ROOT / Path(*rel.parts[1:])
        )

    if len(rel.parts) >= 3:
        candidates.append(
            CHEX_ROOT / Path(*rel.parts[-3:])
        )

    for p in candidates:
        if p.exists():
            return p

    # fallback: match the full relative suffix rather than basename only
    matches = list(
        CHEX_ROOT.rglob(rel.name)
    )

    for p in matches:
        if str(p).replace("\\", "/").endswith(
            str(rel).replace("\\", "/")
        ):
            return p

    if len(matches) == 1:
        return matches[0]

    return None


def load_chex():
    classes = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "pleural effusion",
    ]

    samples = []

    fp = CHEX_ROOT / "chexpert_5x200.csv"

    with fp.open(newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            if row[0].lower() == "path":
                continue

            if len(row) < 6:
                continue

            vals = row[1:6]

            try:
                vals = [int(float(x)) for x in vals]
            except ValueError:
                continue

            positive = [
                i for i, x in enumerate(vals)
                if x == 1
            ]

            if len(positive) != 1:
                continue

            p = resolve_chex_path(row[0])

            if p is None:
                continue

            samples.append(
                (str(p), positive[0])
            )

    return classes, samples


# ============================================================
# METRICS
# ============================================================

def summarize(y_true, scores, n_classes):
    pred = scores.argmax(dim=1)

    recalls = []

    for i in range(n_classes):
        mask = y_true == i

        if mask.sum() == 0:
            recalls.append(float("nan"))
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

    return ba, acc, recalls, counts


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
        num_workers=4
    )

    y_all = []
    single_all = []
    ensemble_all = []

    for images, labels in loader:

        single = get_scores(
            images,
            classes,
            [TEMPLATES[0]],
            clip_model,
            cda,
            device
        )

        ensemble = get_scores(
            images,
            classes,
            TEMPLATES,
            clip_model,
            cda,
            device
        )

        y_all.append(labels.cpu())
        single_all.append(single.cpu())
        ensemble_all.append(ensemble.cpu())

    y_true = torch.cat(y_all)
    single_scores = torch.cat(single_all)
    ensemble_scores = torch.cat(ensemble_all)

    s_ba, s_acc, s_rec, s_counts = summarize(
        y_true,
        single_scores,
        len(classes)
    )

    e_ba, e_acc, e_rec, e_counts = summarize(
        y_true,
        ensemble_scores,
        len(classes)
    )

    print(
        f"Single BA={100*s_ba:.2f}%  "
        f"ACC={100*s_acc:.2f}%"
    )
    print(
        "Single predicted:",
        dict(zip(classes, s_counts))
    )

    print(
        f"7P BA={100*e_ba:.2f}%  "
        f"ACC={100*e_acc:.2f}%"
    )
    print(
        "7P predicted:",
        dict(zip(classes, e_counts))
    )

    return {
        "dataset": name,
        "n": len(samples),
        "single_ba": 100 * s_ba,
        "ensemble_ba": 100 * e_ba,
        "single_acc": 100 * s_acc,
        "ensemble_acc": 100 * e_acc,
        "single_pred_counts": str(
            dict(zip(classes, s_counts))
        ),
        "ensemble_pred_counts": str(
            dict(zip(classes, e_counts))
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", CKPT)

    clip_model, preprocess = clip.load(
        "ViT-B/16",
        device=device,
        jit=False
    )

    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    obj = torch.load(
        CKPT,
        map_location="cpu"
    )

    state = obj["cda_state_dict"]

    cda = CDA(
        d=512,
        h=8,
        d_k=64,
        num_layers=1,
        pool="cls",
        use_proj_heads=True,
    ).to(device)

    cda.load_state_dict(
        state,
        strict=True
    )

    cda.eval()

    datasets = [
        ("ChestX-ray",) + load_chest(),
        ("SIIM-ACR",) + load_siim(),
        ("INbreast",) + load_inbreast(),
        ("CheXpert5x200",) + load_chex(),
    ]

    results = []

    for name, classes, samples in datasets:

        if len(samples) == 0:
            raise RuntimeError(
                f"{name}: zero samples found"
            )

        results.append(
            evaluate_dataset(
                name,
                classes,
                samples,
                clip_model,
                cda,
                preprocess,
                device
            )
        )

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
    print("FINAL TABLE — PAIRWISE + EOT")
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
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
