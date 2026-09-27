
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

from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

FEATURES = (
    ROOT
    / "results/domain_gap/domain_gap_features_seed42.npz"
)

OUT = (
    ROOT
    / "results/domain_gap/domain_gap_verified_metrics.csv"
)

SEED = 42
N_PERM = 199
N_FOLDS = 5
RIDGE = 1.0


# ============================================================
# BASIC MATH
# ============================================================

def sqdist(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True).T

    d2 = xx + yy - 2.0 * (x @ y.T)

    return np.maximum(d2, 0.0)


def estimate_sigma2(feature_sets):
    z = np.concatenate(
        feature_sets,
        axis=0
    ).astype(np.float32)

    d2 = sqdist(z, z)

    tri = d2[
        np.triu_indices(
            len(z),
            k=1
        )
    ]

    tri = tri[tri > 1e-12]

    sigma2 = float(
        np.median(tri)
    )

    if sigma2 <= 0:
        raise RuntimeError(
            f"Invalid sigma2={sigma2}"
        )

    return sigma2


def kernel_matrix(x, y, sigma2):
    z = np.concatenate([x, y], axis=0)

    d2 = sqdist(z, z)

    K = np.exp(
        -d2 / (2.0 * sigma2)
    ).astype(np.float32)

    return K


def mmd_from_indices(
    K,
    idx_x,
    idx_y
):
    Kxx = K[np.ix_(idx_x, idx_x)]
    Kyy = K[np.ix_(idx_y, idx_y)]
    Kxy = K[np.ix_(idx_x, idx_y)]

    return float(
        Kxx.mean()
        + Kyy.mean()
        - 2.0 * Kxy.mean()
    )


def permutation_mmd(
    x,
    y,
    sigma2,
    n_perm,
    seed
):
    n = len(x)
    m = len(y)

    K = kernel_matrix(
        x,
        y,
        sigma2
    )

    idx_x = np.arange(n)
    idx_y = np.arange(n, n + m)

    observed = mmd_from_indices(
        K,
        idx_x,
        idx_y
    )

    rng = np.random.default_rng(seed)

    null = np.empty(
        n_perm,
        dtype=np.float64
    )

    all_idx = np.arange(n + m)

    for b in range(n_perm):
        perm = rng.permutation(all_idx)

        px = perm[:n]
        py = perm[n:]

        null[b] = mmd_from_indices(
            K,
            px,
            py
        )

    p = (
        1
        + np.sum(null >= observed)
    ) / (
        n_perm + 1
    )

    return (
        observed,
        float(p),
        float(null.mean()),
        float(null.std(ddof=1)),
    )


# ============================================================
# LINEAR DOMAIN PROBE
# Pure NumPy ridge classifier
# ============================================================

def auc_rank(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    order = np.argsort(scores)

    sorted_scores = scores[order]

    ranks = np.empty(
        len(scores),
        dtype=float
    )

    i = 0

    while i < len(scores):
        j = i + 1

        while (
            j < len(scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        avg_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    pos = y_true == 1

    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos

    rank_sum = ranks[pos].sum()

    auc = (
        rank_sum
        - n_pos * (n_pos + 1) / 2.0
    ) / (
        n_pos * n_neg
    )

    return float(auc)


def balanced_accuracy(
    y_true,
    y_pred
):
    recalls = []

    for c in [0, 1]:
        mask = y_true == c

        recalls.append(
            np.mean(
                y_pred[mask] == c
            )
        )

    return float(
        np.mean(recalls)
    )


def make_stratified_folds(
    y,
    n_folds,
    seed
):
    rng = np.random.default_rng(seed)

    class_parts = {}

    for c in [0, 1]:
        idx = np.where(y == c)[0]

        rng.shuffle(idx)

        class_parts[c] = np.array_split(
            idx,
            n_folds
        )

    folds = []

    for f in range(n_folds):
        test_idx = np.concatenate(
            [
                class_parts[0][f],
                class_parts[1][f],
            ]
        )

        train_idx = np.setdiff1d(
            np.arange(len(y)),
            test_idx
        )

        folds.append(
            (train_idx, test_idx)
        )

    return folds


def ridge_probe(
    x,
    y,
    n_folds=5,
    ridge=1.0,
    seed=42
):
    X = np.concatenate(
        [x, y],
        axis=0
    ).astype(np.float64)

    domain = np.concatenate(
        [
            np.zeros(len(x), dtype=int),
            np.ones(len(y), dtype=int),
        ]
    )

    target = (
        2.0 * domain - 1.0
    )

    folds = make_stratified_folds(
        domain,
        n_folds,
        seed
    )

    bas = []
    aucs = []

    for train_idx, test_idx in folds:

        Xtr = X[train_idx]
        Xte = X[test_idx]

        ytr = target[train_idx]
        yte = domain[test_idx]

        mu = Xtr.mean(
            axis=0,
            keepdims=True
        )

        sd = Xtr.std(
            axis=0,
            keepdims=True
        )

        sd[sd < 1e-8] = 1.0

        Xtr = (
            Xtr - mu
        ) / sd

        Xte = (
            Xte - mu
        ) / sd

        # Add intercept
        Xtr = np.concatenate(
            [
                Xtr,
                np.ones(
                    (len(Xtr), 1)
                )
            ],
            axis=1
        )

        Xte = np.concatenate(
            [
                Xte,
                np.ones(
                    (len(Xte), 1)
                )
            ],
            axis=1
        )

        d = Xtr.shape[1]

        reg = np.eye(d) * ridge

        # Do not regularize intercept
        reg[-1, -1] = 0.0

        beta = np.linalg.solve(
            Xtr.T @ Xtr + reg,
            Xtr.T @ ytr
        )

        scores = Xte @ beta

        pred = (
            scores > 0
        ).astype(int)

        ba = balanced_accuracy(
            yte,
            pred
        )

        auc = auc_rank(
            yte,
            scores
        )

        bas.append(
            100.0 * ba
        )

        aucs.append(
            100.0 * auc
        )

    return (
        float(np.mean(bas)),
        float(np.std(bas, ddof=1)),
        float(np.mean(aucs)),
        float(np.std(aucs, ddof=1)),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not FEATURES.exists():
        raise FileNotFoundError(
            FEATURES
        )

    z = np.load(
        FEATURES
    )

    expected_keys = [
        "cifar10_a",
        "cifar10_b",
        "ChestXray",
        "SIIM",
        "INbreast",
        "CheXpert5x200",
    ]

    print("Feature file:", FEATURES)
    print("Keys:", list(z.keys()))

    for k in expected_keys:
        if k not in z:
            raise RuntimeError(
                f"Missing feature key: {k}"
            )

        print(
            k,
            z[k].shape
        )

        if z[k].shape != (500, 512):
            raise RuntimeError(
                f"{k}: unexpected shape "
                f"{z[k].shape}"
            )

    feature_sets = [
        z[k]
        for k in expected_keys
    ]

    print()
    print(
        "Estimating common RBF bandwidth..."
    )

    sigma2 = estimate_sigma2(
        feature_sets
    )

    print(
        f"sigma^2 = {sigma2:.9f}"
    )

    comparisons = [
        (
            "CIFAR10_A vs CIFAR10_B",
            z["cifar10_a"],
            z["cifar10_b"],
        ),
        (
            "CIFAR10 vs ChestXray",
            z["cifar10_a"],
            z["ChestXray"],
        ),
        (
            "CIFAR10 vs SIIM",
            z["cifar10_a"],
            z["SIIM"],
        ),
        (
            "CIFAR10 vs INbreast",
            z["cifar10_a"],
            z["INbreast"],
        ),
        (
            "CIFAR10 vs CheXpert5x200",
            z["cifar10_a"],
            z["CheXpert5x200"],
        ),
    ]

    rows = []

    print()
    print("=" * 80)
    print("VERIFIED DOMAIN-GAP RESULTS")
    print("=" * 80)

    for i, (
        name,
        x,
        y
    ) in enumerate(comparisons):

        print()
        print(name)

        (
            mmd2,
            p,
            null_mean,
            null_sd,
        ) = permutation_mmd(
            x,
            y,
            sigma2,
            N_PERM,
            SEED + i
        )

        (
            ba_mean,
            ba_sd,
            auc_mean,
            auc_sd,
        ) = ridge_probe(
            x,
            y,
            n_folds=N_FOLDS,
            ridge=RIDGE,
            seed=SEED,
        )

        print(
            f"MMD^2 = {mmd2:.6f}"
        )

        print(
            f"Permutation p = {p:.4f}"
        )

        print(
            f"Null MMD^2 = "
            f"{null_mean:.6f} "
            f"± {null_sd:.6f}"
        )

        print(
            f"Linear probe BA = "
            f"{ba_mean:.2f} "
            f"± {ba_sd:.2f}%"
        )

        print(
            f"Linear probe AUROC = "
            f"{auc_mean:.2f} "
            f"± {auc_sd:.2f}%"
        )

        rows.append(
            {
                "comparison": name,
                "n_per_domain": len(x),
                "rbf_sigma2": sigma2,
                "mmd2": mmd2,
                "permutation_p":
                    p,
                "null_mmd2_mean":
                    null_mean,
                "null_mmd2_sd":
                    null_sd,
                "probe_BA_mean_pct":
                    ba_mean,
                "probe_BA_sd_pct":
                    ba_sd,
                "probe_AUROC_mean_pct":
                    auc_mean,
                "probe_AUROC_sd_pct":
                    auc_sd,
                "n_permutations":
                    N_PERM,
                "probe":
                    "5-fold ridge linear classifier",
            }
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUT.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    rows[0].keys()
                )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
