from pathlib import Path
import os
import sys
import csv
import math
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
GLORIA_ROOT = PROJECT / "external" / "gloria"

# 让 Python 能找到官方 GLoRIA 代码
sys.path.insert(0, str(GLORIA_ROOT.resolve()))

# 注意：前面我们为了避免 training 依赖，改过 __init__.py
# 所以这里直接从 gloria.gloria 导入官方 zero-shot API
from gloria.gloria import load_gloria, get_similarities

PROMPTS = [
    "This is an image of {c}.",
    "A chest X-ray showing {c}.",
    "Radiograph with findings consistent with {c}.",
    "A medical scan demonstrating {c}.",
    "Imaging study of a patient with {c}.",
    "An X-ray of a patient diagnosed with {c}.",
    "{c} visible on this image."
]


def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]


def build_basename_index(root: Path):
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            idx[p.name] = str(p)
    return idx


def balanced_accuracy_and_recalls(y_true, y_pred, n_cls):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recalls = []
    for c in range(n_cls):
        mask = (y_true == c)
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float((y_pred[mask] == c).mean() * 100.0))

    acc = float((y_true == y_pred).mean() * 100.0)
    ba = float(np.mean(recalls))
    return ba, acc, recalls


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, "values"):
        return x.values
    return np.asarray(x)


def load_chestxray():
    root = PROJECT / "data" / "testsets" / "chest_xray_extracted" / "chest_xray"
    classes = ["normal", "pneumonia"]
    samples = []

    for split in ["train", "val", "test"]:
        for p in sorted((root / split / "NORMAL").glob("*")):
            if p.is_file() and is_image_file(p):
                samples.append((str(p), 0))

        for p in sorted((root / split / "PNEUMONIA").glob("*")):
            if p.is_file() and is_image_file(p):
                samples.append((str(p), 1))

    return "ChestXray", classes, samples

def load_siim():
    csv_path = PROJECT / "data" / "testsets" / "siim_acr_extracted" / "SIIM-ACR" / "test_list.csv"
    img_root = PROJECT / "data" / "testsets" / "siim_acr_extracted"
    img_index = build_basename_index(img_root)

    classes = ["normal", "pneumothorax"]
    samples = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            img_name = os.path.basename(row[0].strip())
            label_str = row[1].strip()
            if label_str not in {"0", "1"}:
                continue
            if img_name in img_index:
                samples.append((img_index[img_name], int(label_str)))

    return "SIIM", classes, samples


def load_inbreast():
    test_root = PROJECT / "data" / "testsets" / "inbreast_extracted" / "INbreast" / "test"
    classes = ["normal", "benign", "malignant"]
    samples = []

    for p in sorted(test_root.rglob("*.jpg")):
        name = p.name
        if "_l0_" in name:
            label = 0
        elif "_l1_" in name:
            label = 1
        elif "_l2_" in name:
            label = 2
        else:
            continue
        samples.append((str(p), label))

    return "INbreast", classes, samples


def load_chexpert5x200():
    csv_path = PROJECT / "data" / "testsets" / "chexpert5x200_extracted" / "chexpert5x200" / "chexpert_5x200.csv"
    img_root = PROJECT / "data" / "testsets" / "chexpert5x200_extracted"
    img_index = build_basename_index(img_root)

    label_cols = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    classes = ["cardiomegaly", "edema", "consolidation", "atelectasis", "pleural effusion"]
    samples = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = os.path.basename(row["Path"].strip())
            if img_name not in img_index:
                continue
            vals = [int(row[c]) for c in label_cols]
            label = int(np.argmax(vals))
            samples.append((img_index[img_name], label))

    return "CheXpert5x200", classes, samples


def all_datasets():
    return [
        load_chestxray(),
        load_siim(),
        load_inbreast(),
        load_chexpert5x200(),
    ]


def make_prompt_dict(classes, templates):
    return {c: [t.format(c=c) for t in templates] for c in classes}


def run_one_setting(model, device, classes, samples, templates, batch_size=64):
    """
    Compute GLoRIA similarities batch-by-batch, but perform
    GLoRIA's class-wise normalization ONCE over the full dataset.

    This avoids batch-dependent normalization and NaNs caused by
    zero standard deviation within a small mini-batch.
    """

    prompt_dict = make_prompt_dict(classes, templates)
    processed_txt = model.process_class_prompts(prompt_dict, device)

    paths = [x[0] for x in samples]
    y_true = np.asarray([x[1] for x in samples], dtype=np.int64)

    all_scores = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]

        processed_imgs = model.process_img(batch_paths, device)

        class_scores = []

        for cls_name in classes:
            # GLoRIA official similarity:
            # both = (global similarity + local similarity) / 2
            sims = get_similarities(
                model,
                processed_imgs,
                processed_txt[cls_name],
                similarity_type="both"
            )

            sims = to_numpy(sims)

            # Shape:
            #   single prompt -> [batch, 1]
            #   seven prompts -> [batch, 7]
            if sims.ndim == 1:
                sims = sims[:, None]

            # Our standardized ensemble rule:
            # average the seven prompt-level similarities
            cls_score = sims.mean(axis=1)

            class_scores.append(cls_score)

        # [batch, num_classes]
        batch_scores = np.stack(class_scores, axis=1)
        all_scores.append(batch_scores)

    # [N, num_classes]
    raw_scores = np.concatenate(all_scores, axis=0)

    if raw_scores.shape[0] != len(y_true):
        raise RuntimeError(
            f"Score/sample mismatch: {raw_scores.shape[0]} vs {len(y_true)}"
        )

    # -----------------------------------------------------
    # IMPORTANT:
    # reproduce GLoRIA's normalization over the FULL dataset,
    # not separately inside each mini-batch.
    # -----------------------------------------------------
    score_mean = raw_scores.mean(axis=0, keepdims=True)
    score_std = raw_scores.std(axis=0, keepdims=True)

    print("Full-dataset score std:",
          [round(float(x), 6) for x in score_std.squeeze()])

    if not np.all(np.isfinite(score_std)):
        raise RuntimeError("Non-finite standard deviation detected.")

    if np.any(score_std == 0):
        raise RuntimeError(
            "A class has zero score standard deviation over the full dataset."
        )

    scores = (raw_scores - score_mean) / score_std

    if not np.all(np.isfinite(scores)):
        raise RuntimeError("NaN/Inf detected after full-dataset normalization.")

    y_pred = scores.argmax(axis=1)

    ba, acc, recalls = balanced_accuracy_and_recalls(
        y_true,
        y_pred,
        len(classes)
    )

    return ba, acc, recalls


def main():
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    out_dir = PROJECT / "results" / "baseline_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"gloria_results_{job_id}.csv"

    # 关键：切到官方仓库目录，确保它按默认相对路径找到 ./pretrained/chexpert_resnet50.ckpt
    os.chdir(GLORIA_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("GLoRIA evaluation")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)

    model = load_gloria(device=device)
    print("GLoRIA model loaded.")

    rows = []

    for dataset_name, classes, samples in all_datasets():
        print()
        print("=" * 70)
        print("Dataset:", dataset_name)
        print("Classes:", classes)
        print("N:", len(samples))
        print("=" * 70)

        single_ba, single_acc, single_recalls = run_one_setting(
            model, device, classes, samples, [PROMPTS[0]]
        )
        print()
        print("Single prompt")
        print("BA:", round(single_ba, 4))
        print("Recalls:", [round(x, 2) for x in single_recalls])

        ens_ba, ens_acc, ens_recalls = run_one_setting(
            model, device, classes, samples, PROMPTS
        )
        print()
        print("7-prompt ensemble")
        print("BA:", round(ens_ba, 4))
        print("Recalls:", [round(x, 2) for x in ens_recalls])
        print("7P - Single:", round(ens_ba - single_ba, 4))

        rows.append({
            "Model": "gloria",
            "Dataset": dataset_name,
            "N": len(samples),
            "Single_BA": round(single_ba, 4),
            "Ensemble7_BA": round(ens_ba, 4),
            "Single_Accuracy": round(single_acc, 4),
            "Ensemble7_Accuracy": round(ens_acc, 4),
            "Single_Recalls": ";".join([f"{x:.4f}" for x in single_recalls]),
            "Ensemble7_Recalls": ";".join([f"{x:.4f}" for x in ens_recalls]),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model", "Dataset", "N",
                "Single_BA", "Ensemble7_BA",
                "Single_Accuracy", "Ensemble7_Accuracy",
                "Single_Recalls", "Ensemble7_Recalls"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("GLORIA FINISHED")
    print("Results:", out_csv)
    print("=" * 70)


if __name__ == "__main__":
    main()
