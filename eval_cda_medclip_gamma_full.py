import argparse
from pathlib import Path

import eval_cda_medclip_baseline_protocol as base

from train_cda_medclip import (
    cda_forward_medclip as original_cda_forward,
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

    # Point baseline-protocol evaluator
    # to this gamma's checkpoint/output

    base.CDA_CKPT = Path(
        args.ckpt
    ).resolve()

    base.OUT_CSV = Path(
        args.out
    ).resolve()


    def gamma_forward(
        cda,
        h_v,
        h_t,
        attention_mask,
    ):

        # original MedCLIP x
        v0 = h_v.mean(
            dim=1
        )

        t0 = h_t.mean(
            dim=1
        )

        if gamma == 0.0:

            return (
                v0,
                t0,
            )

        # CDA(x)
        v_cda, t_cda = (
            original_cda_forward(
                cda,
                h_v,
                h_t,
                attention_mask,
            )
        )

        # output = (1-gamma)x + gamma*CDA(x)

        v = (
            (1.0 - gamma) * v0
            + gamma * v_cda
        )

        t = (
            (1.0 - gamma) * t0
            + gamma * t_cda
        )

        return (
            v,
            t,
        )


    # Replace only CDA fusion rule.
    # External datasets/prompts/metrics remain the same.
    base.cda_forward_medclip = (
        gamma_forward
    )

    print("=" * 72)
    print("GAMMA EXTERNAL ZERO-SHOT")
    print("=" * 72)

    print(
        "gamma:",
        gamma,
    )

    print(
        "output = (1-gamma)x + gamma*CDA(x)"
    )

    print(
        "checkpoint:",
        base.CDA_CKPT,
    )

    base.main()


if __name__ == "__main__":
    main()
