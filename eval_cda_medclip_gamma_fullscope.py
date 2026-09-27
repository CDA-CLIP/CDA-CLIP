import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_cda_medclip_baseline_protocol as base

from train_cda_medclip import (
    cda_forward_medclip as original_cda_forward,
)

from baseline_suite.fullscope_data import (
    all_datasets as fullscope_all_datasets,
)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--gamma",
        type=float,
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    args = p.parse_args()

    gamma = args.gamma

    # --------------------------------------------------------
    # Use FULL-SCOPE external datasets:
    # ChestXray = 5856
    # SIIM = 1250
    # INbreast = 6154
    # CheXpert5x200 = 1000
    # --------------------------------------------------------

    base.all_datasets = fullscope_all_datasets

    # Some versions of the evaluator keep the dataset
    # functions inside the imported baseline module.
    if hasattr(base, "baseline"):
        base.baseline.all_datasets = fullscope_all_datasets

    # --------------------------------------------------------
    # Checkpoint / output
    # --------------------------------------------------------

    base.CDA_CKPT = Path(
        args.ckpt
    ).resolve()

    base.OUT_CSV = Path(
        args.out
    ).resolve()

    # --------------------------------------------------------
    # Gamma residual interpolation
    #
    # output = (1-gamma)x + gamma*CDA(x)
    # --------------------------------------------------------

    def gamma_forward(
        cda,
        h_v,
        h_t,
        attention_mask,
    ):

        v0 = h_v.mean(dim=1)
        t0 = h_t.mean(dim=1)

        if gamma == 0.0:
            return v0, t0

        v_cda, t_cda = original_cda_forward(
            cda,
            h_v,
            h_t,
            attention_mask,
        )

        v = (
            (1.0 - gamma) * v0
            + gamma * v_cda
        )

        t = (
            (1.0 - gamma) * t0
            + gamma * t_cda
        )

        return v, t

    base.cda_forward_medclip = gamma_forward

    print("=" * 72)
    print("CDA-MedCLIP GAMMA FULL-SCOPE ZERO-SHOT")
    print("=" * 72)
    print("gamma =", gamma)
    print("checkpoint =", base.CDA_CKPT)
    print("output =", base.OUT_CSV)
    print("datasets = 5856 / 1250 / 6154 / 1000")
    print("evaluation = Single + 7 prompts")
    print("=" * 72)

    base.main()


if __name__ == "__main__":
    main()
