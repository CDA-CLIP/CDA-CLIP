from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

from baseline_suite.run_eval import (
    PROJECT,
    all_datasets,
    make_text_matrices,
    metrics,
    write_result,
)

from pmc_clip.factory import (
    create_model_and_transforms,
    load_checkpoint,
)


CHECKPOINT = PROJECT / "checkpoints/pmc/checkpoint.pt"


def setup_pmc(device):
    print("Building official PMC-CLIP RN50_fusion4...")

    args = SimpleNamespace(
        model="RN50_fusion4",
        pretrained="",
        device=str(device),
        mlm=True,
        crop_scale=0.1,
        hugging_face=True,
    )

    model, _, preprocess = create_model_and_transforms(
        args=args,
        precision="fp32",
        device=device,
    )

    print("Loading checkpoint:")
    print(CHECKPOINT)

    incompatible = load_checkpoint(
        model,
        str(CHECKPOINT),
        strict=False,
    )

    print("Checkpoint loaded.")
    print("Incompatible keys:", incompatible)

    model = model.to(device)
    model.eval()

    # -------------------------------------------------
    # PMC-CLIP text feature
    #
    # Official PMC code obtains the BERT hidden state
    # at the CLS position and applies text_projection.
    # The fusion branch is not needed for CLIP similarity.
    # -------------------------------------------------
    def encode_texts(texts):
        encoded = model.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=model.context_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)

        with torch.no_grad():
            output = model.text_encoder(
                input_ids=input_ids,
                output_attentions=False,
            )

            hidden = output["last_hidden_state"]

            # PubMedBERT CLS token is position 0
            text_features = hidden[:, 0, :] @ model.text_projection

        return text_features.float()


    def encode_images(paths):
        tensors = []

        for p in paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                tensors.append(preprocess(im))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            output = model.encode_image(batch)
            features = output["image_features"]

        return F.normalize(
            features.float(),
            dim=-1,
        )

    return encode_texts, encode_images, 32


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")

    print("=" * 70)
    print("PMC-CLIP evaluation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)

    encode_texts, encode_images, batch_size = setup_pmc(device)

    output_csv = (
        PROJECT
        / "results/baseline_suite"
        / f"pmc_results_{__import__('os').environ.get('SLURM_JOB_ID', 'manual')}.csv"
    )

    for dataset_name, classes, samples in all_datasets():

        print()
        print("=" * 70)
        print("Dataset:", dataset_name)
        print("Classes:", classes)
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

            image_features = encode_images(paths)

            single_sim = image_features @ single_text.T
            ensemble_sim = image_features @ ensemble_text.T

            pred_single = (
                single_sim.argmax(dim=1)
                .cpu()
                .tolist()
            )

            pred_ensemble = (
                ensemble_sim.argmax(dim=1)
                .cpu()
                .tolist()
            )

            y_true.extend(labels)
            y_single.extend(pred_single)
            y_ensemble.extend(pred_ensemble)

            done = min(
                start + batch_size,
                len(samples),
            )

            if done % 500 == 0 or done == len(samples):
                print(
                    f"{dataset_name}: "
                    f"{done}/{len(samples)}"
                )

        single_ba, single_acc, single_recalls = metrics(
            y_true,
            y_single,
            len(classes),
        )

        ensemble_ba, ensemble_acc, ensemble_recalls = metrics(
            y_true,
            y_ensemble,
            len(classes),
        )

        print()
        print("Single prompt")
        print("BA:", round(single_ba, 4))
        print(
            "Recalls:",
            [round(x, 2) for x in single_recalls],
        )

        print()
        print("7-prompt ensemble")
        print("BA:", round(ensemble_ba, 4))
        print(
            "Recalls:",
            [round(x, 2) for x in ensemble_recalls],
        )

        print(
            "7P - Single:",
            round(ensemble_ba - single_ba, 4),
        )

        write_result(
            output_csv,
            "pmcclip",
            dataset_name,
            len(samples),
            single_ba,
            ensemble_ba,
            single_acc,
            ensemble_acc,
            single_recalls,
            ensemble_recalls,
        )

    print()
    print("=" * 70)
    print("PMC-CLIP FINISHED")
    print("Results:", output_csv)
    print("=" * 70)


if __name__ == "__main__":
    main()
