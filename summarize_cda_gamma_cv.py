from pathlib import Path
import csv
import statistics
from collections import defaultdict


ROOT = Path(
    "results/cda_gamma_cv"
)

rows = []

for f in sorted(
    ROOT.glob(
        "gamma_*_fold*.csv"
    )
):

    with f.open() as fp:

        r = next(
            csv.DictReader(fp)
        )

        rows.append(r)


groups = defaultdict(list)

for r in rows:

    gamma = float(
        r["gamma"]
    )

    groups[gamma].append(
        r
    )


summary = []

for gamma in sorted(groups):

    rs = groups[gamma]

    if len(rs) != 3:
        print(
            f"WARNING gamma={gamma}: "
            f"only {len(rs)} folds"
        )
        continue

    losses = [
        float(x["best_val_loss"])
        for x in rs
    ]

    bir1 = [
        float(x["bidirectional_r1"])
        for x in rs
    ]

    epochs = [
        int(x["best_epoch"])
        for x in rs
    ]

    row = {
        "gamma": gamma,
        "folds": len(rs),
        "mean_val_loss": statistics.mean(
            losses
        ),
        "sd_val_loss": statistics.stdev(
            losses
        ),
        "mean_bidirectional_r1":
            statistics.mean(bir1),
        "best_epochs": str(epochs),
    }

    summary.append(row)


summary.sort(
    key=lambda x: x[
        "mean_val_loss"
    ]
)


out = (
    ROOT
    / "gamma_cv_summary.csv"
)

with out.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys(),
    )

    writer.writeheader()
    writer.writerows(summary)


print()
print("=" * 82)
print("GAMMA CV SUMMARY")
print(
    "Selection criterion = "
    "LOWEST mean source validation loss"
)
print("=" * 82)

print(
    f"{'gamma':>8} "
    f"{'mean val loss':>15} "
    f"{'SD':>10} "
    f"{'mean BI R@1':>15} "
    f"{'best epochs':>20}"
)

for r in summary:

    print(
        f"{r['gamma']:>8} "
        f"{r['mean_val_loss']:>15.4f} "
        f"{r['sd_val_loss']:>10.4f} "
        f"{r['mean_bidirectional_r1']:>15.4f} "
        f"{r['best_epochs']:>20}"
    )


print()
print(
    "SELECTED GAMMA =",
    summary[0]["gamma"],
)

print(
    "Saved:",
    out,
)
