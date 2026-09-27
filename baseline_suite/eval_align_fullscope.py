from pathlib import Path
import csv
import torch
import torch.nn.functional as F
from PIL import Image

import run_eval as base
from fullscope_data import all_datasets

from transformers import AlignModel, AutoProcessor


MODEL_ID = "kakaobrain/align-base"
OUT = Path("results/fullscope_baselines/align.csv")


def as_tensor(x):
    if torch.is_tensor(x):
        return x

    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output

    raise TypeError(f"Unexpected ALIGN feature output: {type(x)}")


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    device = torch.device("cuda")

    print("=" * 72)
    print("ALIGN FULL-SCOPE SINGLE + 7P")
    print("model:", MODEL_ID)
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 72)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AlignModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    def encode_texts(texts):

        inp = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        inp = {
            k: v.to(device)
            for k, v in inp.items()
            if torch.is_tensor(v)
        }

        with torch.no_grad():
            z = model.get_text_features(**inp)

        z = as_tensor(z)

        return F.normalize(
            z.float(),
            dim=-1,
        )

    def encode_images(paths):

        images = []

        for p in paths:
            with Image.open(p) as im:
                images.append(
                    im.convert("RGB").copy()
                )

        inp = processor(
            images=images,
            return_tensors="pt",
        )

        pixels = inp["pixel_values"].to(device)

        with torch.no_grad():
            z = model.get_image_features(
                pixel_values=pixels
            )

        z = as_tensor(z)

        return F.normalize(
            z.float(),
            dim=-1,
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for dataset_name, classes, samples in all_datasets():

        print()
        print("=" * 72)
        print(dataset_name)
        print("N =", len(samples))
        print("classes =", classes)
        print("=" * 72)

        single_text, ensemble_text = (
            base.make_text_matrices(
                classes,
                encode_texts,
                device,
            )
        )

        y_true = []
        y_single = []
        y_ensemble = []

        batch_size = 16

        for start in range(
            0,
            len(samples),
            batch_size,
        ):

            block = samples[
                start:start + batch_size
            ]

            paths = [x[0] for x in block]
            labels = [x[1] for x in block]

            img = encode_images(paths)

            sim_single = (
                img @ single_text.T
            )

            sim_ensemble = (
                img @ ensemble_text.T
            )

            y_single.extend(
                sim_single.argmax(
                    dim=1
                ).cpu().tolist()
            )

            y_ensemble.extend(
                sim_ensemble.argmax(
                    dim=1
                ).cpu().tolist()
            )

            y_true.extend(labels)

            done = min(
                start + batch_size,
                len(samples),
            )

            if (
                done % 500 == 0
                or done == len(samples)
            ):
                print(
                    f"{dataset_name}: "
                    f"{done}/{len(samples)}"
                )

        s_ba, s_acc, s_rec = base.metrics(
            y_true,
            y_single,
            len(classes),
        )

        e_ba, e_acc, e_rec = base.metrics(
            y_true,
            y_ensemble,
            len(classes),
        )

        print(
            f"{dataset_name}: "
            f"Single={s_ba:.4f} "
            f"7P={e_ba:.4f}"
        )

        rows.append({
            "Model": "ALIGN",
            "Dataset": dataset_name,
            "N": len(samples),
            "Single_BA": s_ba,
            "Ensemble7_BA": e_ba,
            "Single_Accuracy": s_acc,
            "Ensemble7_Accuracy": e_acc,
            "Single_Recalls": s_rec,
            "Ensemble7_Recalls": e_rec,
        })

    with OUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("FINAL")
    print("=" * 72)

    for r in rows:
        print(
            r["Dataset"],
            r["N"],
            "Single=",
            round(
                r["Single_BA"],
                2,
            ),
            "7P=",
            round(
                r["Ensemble7_BA"],
                2,
            ),
        )

    print("Saved:", OUT)


if __name__ == "__main__":
    main()
