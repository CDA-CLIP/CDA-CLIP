import os
import csv
from pathlib import Path

ROOT = Path(
    "/scratch/joycewyr_root/joycewyr0/dongnid/training_practice"
)

OUTDIR = (
    ROOT
    / "results/fullscope_baselines"
)

JOB = os.environ.get(
    "SLURM_JOB_ID",
    "manual"
)

rows = []


def read_standard(path, display_name):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append({
                "Model": display_name,
                "Dataset": r["Dataset"],
                "N": int(r["N"]),
                "Single_BA": float(
                    r["Single_BA"]
                ),
                "Ensemble7_BA": float(
                    r["Ensemble7_BA"]
                ),
            })


read_standard(
    OUTDIR / "clip.csv",
    "CLIP"
)

read_standard(
    OUTDIR / "biomedclip.csv",
    "BiomedCLIP"
)

read_standard(
    OUTDIR / "biovil.csv",
    "BioViL"
)

read_standard(
    OUTDIR / "medclip.csv",
    "MedCLIP"
)

read_standard(
    ROOT
    / "results/baseline_suite"
    / f"pmc_results_{JOB}.csv",
    "PMC-CLIP"
)

read_standard(
    ROOT
    / "results/baseline_suite"
    / f"gloria_results_{JOB}.csv",
    "GLoRIA"
)


# Existing Original objective + EOT CDA result
cda_path = (
    ROOT
    / "results"
    / "original_eot_all4_results.csv"
)

with open(
    cda_path,
    newline=""
) as f:

    reader = csv.DictReader(f)

    for r in reader:
        rows.append({
            "Model":
                "CDA-CLIP Original+EOT",

            "Dataset":
                r["dataset"],

            "N":
                int(r["n"]),

            "Single_BA":
                float(r["single_ba"]),

            "Ensemble7_BA":
                float(r["ensemble_ba"]),
        })


# Normalize dataset naming
rename = {
    "ChestX-ray": "ChestXray",
    "SIIM-ACR": "SIIM",
}

for r in rows:
    r["Dataset"] = rename.get(
        r["Dataset"],
        r["Dataset"]
    )


models = [
    "CLIP",
    "BiomedCLIP",
    "BioViL",
    "MedCLIP",
    "PMC-CLIP",
    "GLoRIA",
    "CDA-CLIP Original+EOT",
]

datasets = [
    "ChestXray",
    "SIIM",
    "INbreast",
    "CheXpert5x200",
]


lookup = {
    (
        r["Model"],
        r["Dataset"]
    ): r
    for r in rows
}


summary_csv = (
    OUTDIR
    / f"fullscope_summary_{JOB}.csv"
)

with open(
    summary_csv,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Chest_Single",
        "Chest_7P",
        "SIIM_Single",
        "SIIM_7P",
        "INbreast_Single",
        "INbreast_7P",
        "CheXpert_Single",
        "CheXpert_7P",
    ])

    for model in models:

        vals = [model]

        for ds in datasets:

            r = lookup.get(
                (model, ds)
            )

            if r is None:
                vals += ["", ""]
            else:
                vals += [
                    f"{r['Single_BA']:.2f}",
                    f"{r['Ensemble7_BA']:.2f}",
                ]

        writer.writerow(vals)


print()
print("=" * 118)
print("FULL-SCOPE BALANCED ACCURACY (%)")
print("=" * 118)

header = (
    f"{'Model':<25}"
    f"{'Chest S':>10}"
    f"{'Chest 7P':>10}"
    f"{'SIIM S':>10}"
    f"{'SIIM 7P':>10}"
    f"{'INB S':>10}"
    f"{'INB 7P':>10}"
    f"{'CheX S':>10}"
    f"{'CheX 7P':>10}"
)

print(header)
print("-" * 118)

for model in models:

    vals = []

    for ds in datasets:
        r = lookup.get(
            (model, ds)
        )

        if r is None:
            vals.extend(
                [float("nan"),
                 float("nan")]
            )
        else:
            vals.extend([
                r["Single_BA"],
                r["Ensemble7_BA"],
            ])

    print(
        f"{model:<25}"
        + "".join(
            f"{x:>10.2f}"
            for x in vals
        )
    )

print()
print(
    "Saved summary:",
    summary_csv
)
