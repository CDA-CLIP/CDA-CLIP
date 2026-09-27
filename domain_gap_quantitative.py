import sys
import csv
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

from torchvision.datasets import CIFAR10

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT / "baseline_suite")
)

from fullscope_data import (
    load_chest,
    load_siim,
    load_inbreast,
    load_chexpert,
)

SEED = 42
N_PER_DOMAIN = 500
BATCH_SIZE = 64

CIFAR_ROOT = (
    ROOT
    / "data"
    / "domain_gap"
    / "cifar10"
)

OUT_DIR = (
    ROOT
    / "results"
    / "domain_gap"
)

OUT_CSV = (
    OUT_DIR
    / "domain_gap_metrics.csv"
)

FEATURE_NPZ = (
    OUT_DIR
    / "domain_gap_features_seed42.npz"
)

SAMPLE_CSV = (
    OUT_DIR
    / "domain_gap_selected_samples.csv"
)

EXPECTED_FULL_N = {
    "ChestXray": 5856,
    "SIIM": 1250,
    "INbreast": 6154,
    "CheXpert5x200": 1000,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def allocate_per_class(
    class_labels,
    n_total,
):
    classes = sorted(set(class_labels))
    n_classes = len(classes)

    base = n_total // n_classes
    remainder = n_total % n_classes

    allocation = {}

    for i, c in enumerate(classes):
        allocation[c] = (
            base
            + (1 if i < remainder else 0)
        )

    return allocation


def stratified_sample_pairs(
    samples,
    n_total,
    seed,
):
    groups = defaultdict(list)

    for path, label in samples:
        groups[int(label)].append(
            (path, int(label))
        )

    allocation = allocate_per_class(
        list(groups.keys()),
        n_total,
    )

    rng = random.Random(seed)

    selected = []

    for label in sorted(groups):
        group = list(groups[label])

        need = allocation[label]

        if len(group) < need:
            raise RuntimeError(
                f"Class {label}: need {need}, "
                f"only {len(group)} available"
            )

        rng.shuffle(group)

        selected.extend(
            group[:need]
        )

    rng.shuffle(selected)

    return selected


def stratified_cifar_indices(
    dataset,
    n_total,
    seed,
    exclude=None,
):
    exclude = set(
        [] if exclude is None else exclude
    )

    groups = defaultdict(list)

    for idx, label in enumerate(
        dataset.targets
    ):
        if idx in exclude:
            continue

        groups[int(label)].append(idx)

    allocation = allocate_per_class(
        list(groups.keys()),
        n_total,
    )

    rng = random.Random(seed)

    selected = []

    for label in sorted(groups):
        group = list(groups[label])

        need = allocation[label]

        if len(group) < need:
            raise RuntimeError(
                f"CIFAR class {label}: "
                f"need {need}, "
                f"only {len(group)} available"
            )

        rng.shuffle(group)

        selected.extend(
            group[:need]
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# DATASETS
# ============================================================

class MedicalDataset(Dataset):
    def __init__(
        self,
        samples,
        preprocess,
    ):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.preprocess(
            image
        )

        return image, int(label)


class CIFARSubset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        preprocess,
    ):
        self.dataset = dataset
        self.indices = indices
        self.preprocess = preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image, label = self.dataset[
            real_idx
        ]

        image = self.preprocess(
            image
        )

        return image, int(label)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    dataset,
    model,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    features = []
    labels = []

    for images, y in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        z = model.encode_image(
            images
        )

        z = F.normalize(
            z.float(),
            dim=-1,
        )

        features.append(
            z.cpu().numpy()
        )

        labels.append(
            y.numpy()
        )

    return (
        np.concatenate(
            features,
            axis=0,
        ),
        np.concatenate(
            labels,
            axis=0,
        ),
    )


# ============================================================
# MMD
# ============================================================

def estimate_global_sigma2(
    feature_sets,
):
    z = np.concatenate(
        feature_sets,
        axis=0,
    ).astype(np.float32)

    d2 = euclidean_distances(
        z,
        z,
        squared=True,
    )

    tri = d2[
        np.triu_indices(
            len(z),
            k=1,
        )
    ]

    tri = tri[
        tri > 1e-12
    ]

    sigma2 = float(
        np.median(tri)
    )

    if not np.isfinite(sigma2):
        raise RuntimeError(
            "Invalid RBF bandwidth"
        )

    if sigma2 <= 0:
        raise RuntimeError(
            f"Non-positive sigma2: "
            f"{sigma2}"
        )

    return sigma2


def rbf_mmd2(
    x,
    y,
    sigma2,
):
    d_xx = euclidean_distances(
        x,
        x,
        squared=True,
    )

    d_yy = euclidean_distances(
        y,
        y,
        squared=True,
    )

    d_xy = euclidean_distances(
        x,
        y,
        squared=True,
    )

    k_xx = np.exp(
        -d_xx
        / (2.0 * sigma2)
    )

    k_yy = np.exp(
        -d_yy
        / (2.0 * sigma2)
    )

    k_xy = np.exp(
        -d_xy
        / (2.0 * sigma2)
    )

    # Biased MMD^2 estimator:
    # non-negative and easy to interpret
    mmd2 = (
        k_xx.mean()
        + k_yy.mean()
        - 2.0 * k_xy.mean()
    )

    return float(mmd2)


# ============================================================
# LINEAR DOMAIN PROBE
# ============================================================

def domain_probe(
    x,
    y,
):
    X = np.concatenate(
        [x, y],
        axis=0,
    )

    domain = np.concatenate(
        [
            np.zeros(
                len(x),
                dtype=int,
            ),
            np.ones(
                len(y),
                dtype=int,
            ),
        ]
    )

    clf = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scores = cross_validate(
        clf,
        X,
        domain,
        cv=cv,
        scoring={
            "ba":
                "balanced_accuracy",
            "auc":
                "roc_auc",
        },
        n_jobs=1,
    )

    ba = (
        scores["test_ba"]
        * 100.0
    )

    auc = (
        scores["test_auc"]
        * 100.0
    )

    return {
        "ba_mean":
            float(ba.mean()),

        "ba_sd":
            float(ba.std(
                ddof=1
            )),

        "auc_mean":
            float(auc.mean()),

        "auc_sd":
            float(auc.std(
                ddof=1
            )),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_samples():

    cifar = CIFAR10(
        root=str(CIFAR_ROOT),
        train=False,
        download=True,
    )

    cifar_a = stratified_cifar_indices(
        cifar,
        N_PER_DOMAIN,
        SEED,
    )

    cifar_b = stratified_cifar_indices(
        cifar,
        N_PER_DOMAIN,
        SEED + 1,
        exclude=cifar_a,
    )

    medical = {}

    loaders = [
        load_chest,
        load_siim,
        load_inbreast,
        load_chexpert,
    ]

    for loader in loaders:

        (
            name,
            classes,
            samples,
        ) = loader()

        if (
            name not in
            EXPECTED_FULL_N
        ):
            raise RuntimeError(
                f"Unexpected dataset: "
                f"{name}"
            )

        if (
            len(samples)
            != EXPECTED_FULL_N[name]
        ):
            raise RuntimeError(
                f"{name}: expected full "
                f"N={EXPECTED_FULL_N[name]}, "
                f"got {len(samples)}"
            )

        chosen = (
            stratified_sample_pairs(
                samples,
                N_PER_DOMAIN,
                SEED,
            )
        )

        medical[name] = {
            "classes": classes,
            "samples": chosen,
            "full_n": len(samples),
        }

    return (
        cifar,
        cifar_a,
        cifar_b,
        medical,
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight():

    (
        cifar,
        cifar_a,
        cifar_b,
        medical,
    ) = prepare_samples()

    print(
        "CIFAR10 test N =",
        len(cifar),
    )

    print(
        "CIFAR-A N =",
        len(cifar_a),
    )

    print(
        "CIFAR-B N =",
        len(cifar_b),
    )

    overlap = (
        set(cifar_a)
        & set(cifar_b)
    )

    print(
        "CIFAR A/B overlap =",
        len(overlap),
    )

    if overlap:
        raise RuntimeError(
            "CIFAR control sets overlap"
        )

    for name, obj in medical.items():

        labels = [
            y
            for _, y
            in obj["samples"]
        ]

        print(
            f"{name}: "
            f"full N={obj['full_n']}, "
            f"selected N="
            f"{len(obj['samples'])}, "
            f"class counts="
            f"{dict(Counter(labels))}"
        )

        if (
            len(obj["samples"])
            != N_PER_DOMAIN
        ):
            raise RuntimeError(
                f"{name}: wrong selected N"
            )

    print()
    print(
        "DOMAIN-GAP PREFLIGHT = PASS"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    set_seed(SEED)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        cifar,
        cifar_a,
        cifar_b,
        medical,
    ) = prepare_samples()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 80
    )

    print(
        "QUANTITATIVE NATURAL-TO-MEDICAL "
        "DOMAIN GAP"
    )

    print(
        "=" * 80
    )

    print(
        "Device:",
        device,
    )

    print(
        "Backbone: frozen CLIP ViT-B/16"
    )

    print(
        "Representation: "
        "L2-normalized 512-D global "
        "image embedding"
    )

    print(
        "N per domain:",
        N_PER_DOMAIN,
    )

    model, preprocess = clip.load(
        "ViT-B/16",
        device=device,
        jit=False,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # EXTRACT CIFAR FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Extracting CIFAR10-A ..."
    )

    cifar_a_ds = CIFARSubset(
        cifar,
        cifar_a,
        preprocess,
    )

    feat_cifar_a, y_cifar_a = (
        extract_features(
            cifar_a_ds,
            model,
            device,
        )
    )

    print(
        "CIFAR10-A:",
        feat_cifar_a.shape,
    )

    print(
        "Extracting CIFAR10-B ..."
    )

    cifar_b_ds = CIFARSubset(
        cifar,
        cifar_b,
        preprocess,
    )

    feat_cifar_b, y_cifar_b = (
        extract_features(
            cifar_b_ds,
            model,
            device,
        )
    )

    print(
        "CIFAR10-B:",
        feat_cifar_b.shape,
    )

    # --------------------------------------------------------
    # EXTRACT MEDICAL FEATURES
    # --------------------------------------------------------

    medical_features = {}

    for name, obj in medical.items():

        print()
        print(
            f"Extracting {name} ..."
        )

        ds = MedicalDataset(
            obj["samples"],
            preprocess,
        )

        feat, labels = extract_features(
            ds,
            model,
            device,
        )

        medical_features[name] = (
            feat,
            labels,
        )

        print(
            f"{name}:",
            feat.shape,
        )

    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    np.savez_compressed(
        FEATURE_NPZ,
        cifar10_a=feat_cifar_a,
        cifar10_b=feat_cifar_b,
        ChestXray=
            medical_features[
                "ChestXray"
            ][0],
        SIIM=
            medical_features[
                "SIIM"
            ][0],
        INbreast=
            medical_features[
                "INbreast"
            ][0],
        CheXpert5x200=
            medical_features[
                "CheXpert5x200"
            ][0],
    )

    # --------------------------------------------------------
    # SAVE SELECTED SAMPLE MANIFEST
    # --------------------------------------------------------

    with SAMPLE_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "domain",
                "sample",
                "original_label",
            ]
        )

        for idx in cifar_a:
            writer.writerow(
                [
                    "CIFAR10_A",
                    f"test_index:{idx}",
                    cifar.targets[idx],
                ]
            )

        for idx in cifar_b:
            writer.writerow(
                [
                    "CIFAR10_B",
                    f"test_index:{idx}",
                    cifar.targets[idx],
                ]
            )

        for name, obj in medical.items():

            for path, label in obj[
                "samples"
            ]:
                writer.writerow(
                    [
                        name,
                        str(path),
                        label,
                    ]
                )

    # --------------------------------------------------------
    # ONE COMMON RBF BANDWIDTH FOR ALL COMPARISONS
    # --------------------------------------------------------

    all_sets = [
        feat_cifar_a,
        feat_cifar_b,
    ]

    for name in [
        "ChestXray",
        "SIIM",
        "INbreast",
        "CheXpert5x200",
    ]:
        all_sets.append(
            medical_features[
                name
            ][0]
        )

    print()
    print(
        "Estimating common RBF "
        "bandwidth ..."
    )

    sigma2 = (
        estimate_global_sigma2(
            all_sets
        )
    )

    print(
        "Common sigma^2 =",
        sigma2,
    )

    # --------------------------------------------------------
    # COMPARISONS
    # --------------------------------------------------------

    comparisons = [
        (
            "CIFAR10_A vs CIFAR10_B",
            feat_cifar_a,
            feat_cifar_b,
        ),
        (
            "CIFAR10 vs ChestXray",
            feat_cifar_a,
            medical_features[
                "ChestXray"
            ][0],
        ),
        (
            "CIFAR10 vs SIIM",
            feat_cifar_a,
            medical_features[
                "SIIM"
            ][0],
        ),
        (
            "CIFAR10 vs INbreast",
            feat_cifar_a,
            medical_features[
                "INbreast"
            ][0],
        ),
        (
            "CIFAR10 vs CheXpert5x200",
            feat_cifar_a,
            medical_features[
                "CheXpert5x200"
            ][0],
        ),
    ]

    results = []

    print()
    print(
        "=" * 80
    )

    print(
        "QUANTITATIVE RESULTS"
    )

    print(
        "=" * 80
    )

    for (
        comparison,
        x,
        y,
    ) in comparisons:

        print()
        print(
            comparison
        )

        mmd2 = rbf_mmd2(
            x,
            y,
            sigma2,
        )

        probe = domain_probe(
            x,
            y,
        )

        row = {
            "comparison":
                comparison,

            "n_per_domain":
                len(x),

            "rbf_sigma2":
                sigma2,

            "mmd2_rbf":
                mmd2,

            "probe_BA_mean_pct":
                probe["ba_mean"],

            "probe_BA_sd_pct":
                probe["ba_sd"],

            "probe_AUROC_mean_pct":
                probe["auc_mean"],

            "probe_AUROC_sd_pct":
                probe["auc_sd"],
        }

        results.append(
            row
        )

        print(
            f"MMD^2 = "
            f"{mmd2:.6f}"
        )

        print(
            f"Probe BA = "
            f"{probe['ba_mean']:.2f}"
            f" ± "
            f"{probe['ba_sd']:.2f}%"
        )

        print(
            f"Probe AUROC = "
            f"{probe['auc_mean']:.2f}"
            f" ± "
            f"{probe['auc_sd']:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    results[0].keys()
                ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        "=" * 80
    )

    print(
        "DOMAIN-GAP ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Metrics:",
        OUT_CSV,
    )

    print(
        "Features:",
        FEATURE_NPZ,
    )

    print(
        "Sample manifest:",
        SAMPLE_CSV,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()
    else:
        main()
