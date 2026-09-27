import os
import csv
import re
from pathlib import Path

PROJECT = Path(
    str(Path(__file__).resolve().parents[1])
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# ChestX-ray: full train + val + test = 5856
# ============================================================

def load_chest():
    root = (
        PROJECT
        / "data/testsets/chest_xray_extracted/chest_xray"
    )

    classes = ["normal", "pneumonia"]
    samples = []

    for split in ["train", "val", "test"]:
        for label, folder in [
            (0, "NORMAL"),
            (1, "PNEUMONIA"),
        ]:
            d = root / split / folder

            for p in sorted(d.glob("*")):
                if (
                    p.is_file()
                    and p.suffix.lower() in IMG_EXT
                ):
                    samples.append(
                        (p, label)
                    )

    return "ChestXray", classes, samples


# ============================================================
# SIIM: full train_list + test_list = 1250
#
# Released CDA mapping:
# 0 = collapsed lung
# 1 = normal lung
# ============================================================

def load_siim():
    root = (
        PROJECT
        / "data/testsets/siim_acr_extracted/SIIM-ACR"
    )

    classes = [
        "collapsed lung",
        "normal lung",
    ]

    image_index = {}

    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in IMG_EXT
        ):
            image_index.setdefault(
                p.name,
                p
            )

    samples = []

    for csv_name in [
        "train_list.csv",
        "test_list.csv",
    ]:
        fp = root / csv_name

        with fp.open(
            newline=""
        ) as f:

            reader = csv.reader(f)

            for row in reader:

                if len(row) < 2:
                    continue

                try:
                    label = int(
                        float(row[1])
                    )
                except Exception:
                    continue

                old_path = row[0].strip()
                candidate = Path(old_path)

                if candidate.exists():
                    p = candidate
                else:
                    p = image_index.get(
                        candidate.name
                    )

                if p is None:
                    raise FileNotFoundError(
                        f"Cannot resolve SIIM: {old_path}"
                    )

                samples.append(
                    (p, label)
                )

    return "SIIM", classes, samples


# ============================================================
# INbreast: full train + test = 6154
# ============================================================

def load_inbreast():
    root = (
        PROJECT
        / "data/testsets/inbreast_extracted/INbreast"
    )

    classes = [
        "normal",
        "benign",
        "malignant",
    ]

    samples = []

    for split in ["train", "test"]:
        d = root / split

        for p in sorted(d.glob("*")):

            if (
                not p.is_file()
                or p.suffix.lower() not in IMG_EXT
            ):
                continue

            m = re.search(
                r"_l([012])",
                p.name
            )

            if m is None:
                continue

            samples.append(
                (
                    p,
                    int(m.group(1))
                )
            )

    return "INbreast", classes, samples


# ============================================================
# CheXpert5x200: full 1000
# ============================================================

def load_chexpert():
    root = (
        PROJECT
        / "data/testsets/"
          "chexpert5x200_extracted/"
          "chexpert5x200"
    )

    csv_path = (
        root
        / "chexpert_5x200.csv"
    )

    classes = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "pleural effusion",
    ]

    samples = []

    with csv_path.open(
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        # Match semantic class names,
        # not basename-only indexing.
        for row in reader:

            label_map = {
                "Atelectasis": 0,
                "Cardiomegaly": 1,
                "Consolidation": 2,
                "Edema": 3,
                "Pleural Effusion": 4,
            }

            positives = []

            for col, idx in label_map.items():
                try:
                    if float(row[col]) == 1.0:
                        positives.append(idx)
                except Exception:
                    pass

            if len(positives) != 1:
                continue

            rel = Path(
                row["Path"].strip()
            )

            candidates = [
                root / rel
            ]

            if len(rel.parts) > 1:
                candidates.append(
                    root
                    / Path(*rel.parts[1:])
                )

            if len(rel.parts) >= 3:
                candidates.append(
                    root
                    / Path(*rel.parts[-3:])
                )

            p = None

            for candidate in candidates:
                if candidate.exists():
                    p = candidate
                    break

            if p is None:
                matches = list(
                    root.rglob(rel.name)
                )

                if len(matches) == 1:
                    p = matches[0]

            if p is None:
                raise FileNotFoundError(
                    f"Cannot resolve CheXpert: {rel}"
                )

            samples.append(
                (p, positives[0])
            )

    return (
        "CheXpert5x200",
        classes,
        samples
    )


def all_datasets():
    datasets = [
        load_chest(),
        load_siim(),
        load_inbreast(),
        load_chexpert(),
    ]

    print()
    print("=" * 70)
    print("FULL-SCOPE DATA AUDIT")
    print("=" * 70)

    expected = {
        "ChestXray": 5856,
        "SIIM": 1250,
        "INbreast": 6154,
        "CheXpert5x200": 1000,
    }

    for name, classes, samples in datasets:
        print(
            name,
            "N=",
            len(samples),
            "classes=",
            classes
        )

        if len(samples) != expected[name]:
            raise RuntimeError(
                f"{name}: expected "
                f"{expected[name]}, "
                f"found {len(samples)}"
            )

    return datasets
