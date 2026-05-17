"""
Pure-Python parameter accounting for the Cross-Domain Attention (CDA)
module defined in CDA.py.

This script is intentionally torch-free: it enumerates every learnable
tensor in the CDA module symbolically (matching the constructors in
CDA.py one-for-one) and prints exact totals for several configurations.

The intention is to give the exact number that should be reported in
the manuscript's complexity table, replacing the previous (incorrect)
12.7 M figure.
"""

from typing import List, Tuple


def linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    """Count parameters of nn.Linear(in_features, out_features, bias)."""
    return in_features * out_features + (out_features if bias else 0)


def layernorm_params(d: int) -> int:
    """nn.LayerNorm has weight (gamma) and bias (beta), each of length d."""
    return 2 * d


def cross_attention_block(d: int = 512, h: int = 8, d_k: int = 64,
                          bias: bool = True) -> List[Tuple[str, int]]:
    """One single-direction multi-head cross-attention block:
       W_Q, W_K, W_V : Linear(d, h*d_k);  W_O : Linear(h*d_k, d).
    """
    assert h * d_k == d
    return [
        ("W_Q", linear_params(d, h * d_k, bias)),
        ("W_K", linear_params(d, h * d_k, bias)),
        ("W_V", linear_params(d, h * d_k, bias)),
        ("W_O", linear_params(h * d_k, d, bias)),
    ]


def cda_layer(d: int = 512, h: int = 8, d_k: int = 64,
              bias: bool = True) -> List[Tuple[str, int]]:
    """One CDA layer: two cross-attention blocks (v<-t and t<-v) and
    the two learnable residual-gating scalars alpha and beta."""
    parts: List[Tuple[str, int]] = []
    for name, n in cross_attention_block(d, h, d_k, bias):
        parts.append((f"v<-t.{name}", n))
    for name, n in cross_attention_block(d, h, d_k, bias):
        parts.append((f"t<-v.{name}", n))
    parts.append(("alpha", 1))
    parts.append(("beta", 1))
    return parts


def cda_module(d: int = 512, h: int = 8, d_k: int = 64,
               num_layers: int = 1, use_proj_heads: bool = True,
               bias: bool = True) -> List[Tuple[str, int]]:
    """Full CDA module: stacked CDA layers + LayerNorms + projection heads."""
    parts: List[Tuple[str, int]] = []
    for L in range(num_layers):
        for name, n in cda_layer(d, h, d_k, bias):
            parts.append((f"layer{L}.{name}", n))
    parts.append(("norm_v", layernorm_params(d)))
    parts.append(("norm_t", layernorm_params(d)))
    if use_proj_heads:
        parts.append(("W_v(proj)", linear_params(d, d, bias)))
        parts.append(("W_t(proj)", linear_params(d, d, bias)))
    return parts


def _fmt(n: int) -> str:
    return f"{n:>13,d}  (= {n/1e6:6.3f} M)"


def print_report(d: int = 512, h: int = 8, d_k: int = 64,
                 num_layers: int = 1, use_proj_heads: bool = True,
                 bias: bool = True, verbose: bool = False) -> int:
    parts = cda_module(d, h, d_k, num_layers, use_proj_heads, bias)
    total = sum(n for _, n in parts)
    if verbose:
        print(f"\nDetailed breakdown (d={d}, h={h}, d_k={d_k}, "
              f"num_layers={num_layers}, proj_heads={use_proj_heads}, bias={bias}):")
        for name, n in parts:
            print(f"   {name:<32s}  {n:>10,d}")
        print(f"   {'TOTAL':<32s}  {total:>10,d}")
    return total


def main() -> None:
    d, h, d_k = 512, 8, 64
    assert h * d_k == d

    print("============================================================")
    print(" Cross-Domain Attention (CDA) parameter audit")
    print(" Configuration from the manuscript: d=512, h=8, d_k=64")
    print(" (so that h * d_k = d, the shared hidden dimension)")
    print("============================================================\n")

    # 1. The paper's analytical formula, sanity check.
    one_direction = 4 * h * d_k * d
    print("Analytical formula 4 h d_k d (one cross-attention direction):")
    print(f"   4 * {h} * {d_k} * {d} = {one_direction:,d}  "
          f"(= {one_direction/1e6:.3f} M)")
    print("   <-- this is what the manuscript's formula computes;")
    print("       it accounts for ONLY ONE of the two cross-attention")
    print("       directions and EXCLUDES projection heads and biases.\n")

    # 2. The complete CDA layer (both directions).
    one_layer_weights_only = 8 * h * d_k * d
    print("Both cross-attention directions in one CDA layer (weights only):")
    print(f"   8 * {h} * {d_k} * {d} = {one_layer_weights_only:,d}  "
          f"(= {one_layer_weights_only/1e6:.3f} M)\n")

    # 3. Full module with stacked depths.
    print("Total trainable parameters of the *implemented* CDA module")
    print("(includes biases, alpha/beta, LayerNorms, projection heads):")
    print(f"   {'CDA layers':<12s}  {'trainable params':>30s}")
    for L in (1, 2, 3, 6):
        total = print_report(d, h, d_k, num_layers=L,
                              use_proj_heads=True, bias=True)
        print(f"   {L:<12d}  {_fmt(total)}")

    print("\nMatching the (incorrect) 12.7 M figure would require ~6 layers;")
    print("the manuscript's text and ablation study, however, only ever")
    print("describe up to 3 CDA layers.  The honest number to report is")
    print("the row for the depth that is actually trained.\n")

    # 4. Verbose breakdown for the default 1-layer configuration.
    print_report(d, h, d_k, num_layers=1, use_proj_heads=True,
                 bias=True, verbose=True)


if __name__ == "__main__":
    main()
