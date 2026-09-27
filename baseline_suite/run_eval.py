import os
import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


PROJECT = Path(os.environ.get("CDA_CLIP_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))

TEMPLATES = [
    "This is an image of {c}.",
    "A chest X-ray showing {c}.",
    "Radiograph with findings consistent with {c}.",
    "A medical scan demonstrating {c}.",
    "Imaging study of a patient with {c}.",
    "An X-ray of a patient diagnosed with {c}.",
    "{c} visible on this image.",
]

TABLE2 = {
    "clip": {
        "ChestXray": 50.00,
        "SIIM": 50.00,
        "INbreast": 33.26,
        "CheXpert5x200": 20.00,
    },
    "medclip": {
        "ChestXray": 59.87,
        "SIIM": 50.41,
        "INbreast": 40.65,
        "CheXpert5x200": 25.00,
    },
    "biomedclip": {
        "ChestXray": 60.34,
        "SIIM": 50.69,
        "INbreast": 41.02,
        "CheXpert5x200": 25.50,
    },
    "biovil": {
        "ChestXray": 57.85,
        "SIIM": 50.00,
        "INbreast": 38.94,
        "CheXpert5x200": 22.80,
    },
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_chest():
    root = PROJECT / "data/testsets/chest_xray_extracted/chest_xray"
    samples = []

    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        parent = p.parent.name.upper()
        if parent == "NORMAL":
            samples.append((p, 0))
        elif parent == "PNEUMONIA":
            samples.append((p, 1))

    return "ChestXray", ["normal", "pneumonia"], samples


def load_siim():
    root = PROJECT / "data/testsets/siim_acr_extracted/SIIM-ACR"

    csvs = list(root.rglob("test_list.csv"))
    if not csvs:
        raise FileNotFoundError("SIIM test_list.csv not found")

    image_index = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            image_index.setdefault(p.name, p)

    samples = []
    with open(csvs[0], newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            old_path = row[0].strip()
            label = int(float(row[1]))

            candidate = Path(old_path)
            if candidate.exists():
                p = candidate
            else:
                p = image_index.get(candidate.name)

            if p is None:
                raise FileNotFoundError(f"Cannot resolve SIIM image: {old_path}")

            samples.append((p, label))

    return "SIIM", ["normal", "pneumothorax"], samples


def load_inbreast():
    root = PROJECT / "data/testsets/inbreast_extracted/INbreast"
    test_root = root / "test"

    samples = []
    for p in sorted(test_root.glob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue

        m = re.search(r"_l([012])", p.name)
        if not m:
            continue

        samples.append((p, int(m.group(1))))

    return "INbreast", ["normal", "benign", "malignant"], samples


def load_chexpert():
    root = PROJECT / "data/testsets/chexpert5x200_extracted/chexpert5x200"

    csvs = list(root.rglob("chexpert_5x200.csv"))
    if not csvs:
        raise FileNotFoundError("chexpert_5x200.csv not found")

    class_cols = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    samples = []

    with open(csvs[0], newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            positives = []
            for i, col in enumerate(class_cols):
                try:
                    if float(row[col]) == 1.0:
                        positives.append(i)
                except Exception:
                    pass

            if len(positives) != 1:
                continue

            rel = row["Path"].strip()
            p = root / rel

            if not p.exists():
                matches = list(root.rglob(Path(rel).name))
                if not matches:
                    raise FileNotFoundError(f"Cannot resolve CheXpert image: {rel}")
                p = matches[0]

            samples.append((p, positives[0]))

    classes = [
        "cardiomegaly",
        "edema",
        "consolidation",
        "atelectasis",
        "pleural effusion",
    ]

    return "CheXpert5x200", classes, samples


def all_datasets():
    return [
        load_chest(),
        load_siim(),
        load_inbreast(),
        load_chexpert(),
    ]


def make_text_matrices(classes, encode_texts, device):
    singles = []
    ensembles = []

    for cls in classes:
        prompts = [x.format(c=cls) for x in TEMPLATES]

        with torch.no_grad():
            features = encode_texts(prompts)

        features = F.normalize(features.float(), dim=-1)

        single = features[0]
        ensemble = F.normalize(features.mean(dim=0), dim=0)

        singles.append(single)
        ensembles.append(ensemble)

    single_matrix = torch.stack(singles).to(device)
    ensemble_matrix = torch.stack(ensembles).to(device)

    single_matrix = F.normalize(single_matrix, dim=-1)
    ensemble_matrix = F.normalize(ensemble_matrix, dim=-1)

    return single_matrix, ensemble_matrix


def metrics(y_true, y_pred, nclass):
    recalls = []

    for c in range(nclass):
        idx = [i for i, y in enumerate(y_true) if y == c]

        if not idx:
            recalls.append(float("nan"))
            continue

        correct = sum(y_pred[i] == c for i in idx)
        recalls.append(correct / len(idx))

    valid = [x for x in recalls if x == x]
    ba = sum(valid) / len(valid)

    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

    return ba * 100.0, acc * 100.0, [x * 100.0 for x in recalls]


def write_result(
    out,
    model_name,
    dataset,
    n,
    single_ba,
    ens_ba,
    single_acc,
    ens_acc,
    single_recalls,
    ens_recalls,
):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    exists = out.exists()

    expected = TABLE2.get(model_name, {}).get(dataset, "")
    delta = ""

    if expected != "":
        delta = single_ba - expected

    with open(out, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "Model",
                "Dataset",
                "N",
                "Single_BA",
                "Ensemble7_BA",
                "Single_Accuracy",
                "Ensemble7_Accuracy",
                "Table2_Single",
                "Single_Delta_vs_Table2",
                "Single_Recalls",
                "Ensemble7_Recalls",
            ])

        writer.writerow([
            model_name,
            dataset,
            n,
            f"{single_ba:.4f}",
            f"{ens_ba:.4f}",
            f"{single_acc:.4f}",
            f"{ens_acc:.4f}",
            expected,
            f"{delta:.4f}" if delta != "" else "",
            ";".join(f"{x:.4f}" for x in single_recalls),
            ";".join(f"{x:.4f}" for x in ens_recalls),
        ])


def open_rgb(paths):
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB").copy())
    return imgs


def setup_clip(device):
    import clip

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    def encode_texts(texts):
        tokens = clip.tokenize(texts).to(device)
        return model.encode_text(tokens)

    def encode_images(paths):
        images = open_rgb(paths)
        x = torch.stack([preprocess(im) for im in images]).to(device)

        with torch.no_grad():
            emb = model.encode_image(x)

        return F.normalize(emb.float(), dim=-1)

    return encode_texts, encode_images, 64


def setup_biomedclip(device):
    import open_clip

    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    def encode_texts(texts):
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens)

    def encode_images(paths):
        images = open_rgb(paths)
        x = torch.stack([preprocess(im) for im in images]).to(device)

        with torch.no_grad():
            emb = model.encode_image(x)

        return F.normalize(emb.float(), dim=-1)

    return encode_texts, encode_images, 48


def setup_medclip(device):
    from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor

    checkpoint = PROJECT / "pretrained/medclip-resnet/pytorch_model.bin"

    if not checkpoint.exists():
        raise FileNotFoundError(f"MedCLIP checkpoint missing: {checkpoint}")

    print("Loading MedCLIP ResNet50 checkpoint:", checkpoint)

    model = MedCLIPModel(vision_cls=MedCLIPVisionModel)

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    processor = MedCLIPProcessor()

    def encode_texts(texts):
        inp = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            emb = model.encode_text(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
            )

        return emb

    def encode_images(paths):
        images = open_rgb(paths)

        inp = processor(
            images=images,
            return_tensors="pt",
        )

        pixels = inp["pixel_values"].to(device)

        with torch.no_grad():
            emb = model.encode_image(pixel_values=pixels)

        return F.normalize(emb.float(), dim=-1)

    return encode_texts, encode_images, 32


def setup_biovil(device):
    from health_multimodal.image.utils import (
        ImageModelType,
        get_image_inference,
    )
    from health_multimodal.text.utils import (
        BertEncoderType,
        get_bert_inference,
    )

    print("Loading BioViL image encoder...")
    image_engine = get_image_inference(ImageModelType.BIOVIL)

    print("Loading CXR-BERT text encoder...")
    text_engine = get_bert_inference(BertEncoderType.CXR_BERT)

    image_engine.to(device)
    text_engine.to(device)

    image_engine.model.eval()
    text_engine.model.eval()

    def encode_texts(texts):
        with torch.no_grad():
            emb = text_engine.get_embeddings_from_prompt(
                texts,
                normalize=True,
                verbose=False,
            )
        return emb

    # BioViL official inference engine processes one image at a time.
    def encode_images(paths):
        rows = []

        for p in paths:
            with torch.no_grad():
                emb = image_engine.get_projected_global_embedding(Path(p))
            rows.append(emb)

        return F.normalize(torch.stack(rows).float(), dim=-1)

    return encode_texts, encode_images, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["clip", "medclip", "biomedclip", "biovil"],
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this overnight evaluation")

    device = torch.device("cuda")

    print("=" * 70)
    print("MODEL:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)

    if args.model == "clip":
        encode_texts, encode_images, batch_size = setup_clip(device)

    elif args.model == "biomedclip":
        encode_texts, encode_images, batch_size = setup_biomedclip(device)

    elif args.model == "medclip":
        encode_texts, encode_images, batch_size = setup_medclip(device)

    elif args.model == "biovil":
        encode_texts, encode_images, batch_size = setup_biovil(device)

    else:
        raise ValueError(args.model)

    for dataset_name, classes, samples in all_datasets():
        print()
        print("=" * 70)
        print(dataset_name)
        print("classes:", classes)
        print("N:", len(samples))
        print("=" * 70)

        single_text, ensemble_text = make_text_matrices(
            classes,
            encode_texts,
            device,
        )

        y_true = []
        y_single = []
        y_ensemble = []

        for start in range(0, len(samples), batch_size):
            block = samples[start:start + batch_size]

            paths = [x[0] for x in block]
            labels = [x[1] for x in block]

            img = encode_images(paths).to(device)
            img = F.normalize(img.float(), dim=-1)

            sim_single = img @ single_text.T
            sim_ensemble = img @ ensemble_text.T

            pred_single = sim_single.argmax(dim=1).cpu().tolist()
            pred_ensemble = sim_ensemble.argmax(dim=1).cpu().tolist()

            y_true.extend(labels)
            y_single.extend(pred_single)
            y_ensemble.extend(pred_ensemble)

            done = min(start + batch_size, len(samples))
            if done % 500 == 0 or done == len(samples):
                print(f"{dataset_name}: {done}/{len(samples)}")

        s_ba, s_acc, s_rec = metrics(
            y_true,
            y_single,
            len(classes),
        )

        e_ba, e_acc, e_rec = metrics(
            y_true,
            y_ensemble,
            len(classes),
        )

        print()
        print("Single prompt:")
        print("  recalls:", [round(x, 2) for x in s_rec])
        print("  accuracy:", round(s_acc, 2))
        print("  balanced accuracy:", round(s_ba, 2))

        print("7-prompt ensemble:")
        print("  recalls:", [round(x, 2) for x in e_rec])
        print("  accuracy:", round(e_acc, 2))
        print("  balanced accuracy:", round(e_ba, 2))

        expected = TABLE2.get(args.model, {}).get(dataset_name)
        if expected is not None:
            print(
                "Table2 single:",
                expected,
                "delta:",
                round(s_ba - expected, 2),
            )

        write_result(
            args.out,
            args.model,
            dataset_name,
            len(samples),
            s_ba,
            e_ba,
            s_acc,
            e_acc,
            s_rec,
            e_rec,
        )

    print()
    print("MODEL FINISHED:", args.model)


if __name__ == "__main__":
    main()
