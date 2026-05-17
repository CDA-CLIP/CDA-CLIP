"""
Cross-Domain Attention (CDA) module.

This file implements the CDA module exactly as described in the manuscript
"Latent Representation Alignment for Medical Multi-Modal Learning"
(Section "Proposed Method: Cross-Domain Attention (CDA)").

Specifically:
  - Visual tokens  h_v in R^{n_v x d}
  - Textual tokens h_t in R^{n_t x d}
  - Two symmetric multi-head cross-attention blocks:
        v <- t : Q_v = h_v W_Q^(v), K_t = h_t W_K^(t), V_t = h_t W_V^(t)
        t <- v : Q_t = h_t W_Q^(t), K_v = h_v W_K^(v), V_v = h_v W_V^(v)
  - Standard multi-head dot-product attention with h heads of dimension d_k
    (so that h * d_k = d), an output projection W_O is applied to the
    concatenated heads (as in the canonical Transformer formulation).
  - Residual fusion with learnable scalars alpha, beta:
        tilde h_v = h_v + alpha * Attn_{v<-t}(h_v, h_t)
        tilde h_t = h_t + beta  * Attn_{t<-v}(h_t, h_v)
  - LayerNorm and modality-specific linear projection heads W_v, W_t
    map the pooled (CLS / mean) tokens into the shared contrastive space.

Default configuration used in the paper:
    h = 8,  d_k = 64,  d = 512,  n_v = 50,  n_t = 77

Parameter accounting per CDA *layer* (one bidirectional block):
    Each direction (Q, K, V, O) =  4 * d * d                    [+ 4*d biases]
    Two directions               =  8 * d * d                   [+ 8*d biases]
    Residual scalars alpha, beta =  2
With d = 512:
    8 * 512^2 = 2,097,152  ≈ 2.10 M  weights (biases ~4 K)

Full CDA module (one layer + 2 projection heads W_v, W_t):
    8 d^2 + 2 d^2 + 2 = 10 d^2 + 2  ≈ 2.62 M  (weights only)

An earlier draft of the manuscript wrote the per-layer parameter count
as "4 h d_k d" -- that formula accounts for only one of the two
cross-attention directions and excludes biases and the projection heads.
The correct count for the configuration described in the paper
(single bidirectional CDA layer with LayerNorms and projection heads)
is computed by `count_cda_parameters()` below.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """A single-direction multi-head cross-attention block.

    Computes  Attn(Q from x_q, K/V from x_kv).  This corresponds to one of
    the two sub-blocks (v<-t or t<-v) of a CDA layer.

    Parameters:
        d        : shared hidden dimension (== h * d_k)
        h        : number of attention heads
        d_k      : per-head dimension (default d // h)
        attn_drop, proj_drop : dropout probabilities
        bias     : whether the linear projections have a bias term
    """

    def __init__(self, d: int = 512, h: int = 8, d_k: Optional[int] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if d_k is None:
            d_k = d // h
        assert h * d_k == d, (
            f"h * d_k ({h*d_k}) must equal d ({d})."
        )
        self.h = h
        self.d_k = d_k
        self.d = d
        self.scale = d_k ** -0.5

        # Q from query modality, K/V from key-value modality.
        self.W_Q = nn.Linear(d, h * d_k, bias=bias)
        self.W_K = nn.Linear(d, h * d_k, bias=bias)
        self.W_V = nn.Linear(d, h * d_k, bias=bias)
        # Standard output projection that mixes the concatenated heads.
        self.W_O = nn.Linear(h * d_k, d, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_q  : (B, n_q, d)   query tokens
        x_kv : (B, n_kv, d)  key/value tokens
        key_padding_mask : (B, n_kv) bool, True for positions to mask out
        Returns: (B, n_q, d)
        """
        B, n_q, _ = x_q.shape
        n_kv = x_kv.shape[1]

        Q = self.W_Q(x_q).view(B, n_q, self.h, self.d_k).transpose(1, 2)   # (B, h, n_q, d_k)
        K = self.W_K(x_kv).view(B, n_kv, self.h, self.d_k).transpose(1, 2) # (B, h, n_kv, d_k)
        V = self.W_V(x_kv).view(B, n_kv, self.h, self.d_k).transpose(1, 2) # (B, h, n_kv, d_k)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale           # (B, h, n_q, n_kv)
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                        # (B, h, n_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, n_q, self.h * self.d_k)
        out = self.W_O(out)
        out = self.proj_drop(out)
        return out


class CDALayer(nn.Module):
    """A single Cross-Domain Attention layer (bidirectional).

    Implements Eqs. (5)-(10) of the manuscript:
        tilde h_v = h_v + alpha * Attn_{v<-t}(h_v, h_t)
        tilde h_t = h_t + beta  * Attn_{t<-v}(h_t, h_v)
    """

    def __init__(self, d: int = 512, h: int = 8, d_k: Optional[int] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.v_from_t = CrossAttentionBlock(d, h, d_k, attn_drop, proj_drop, bias)
        self.t_from_v = CrossAttentionBlock(d, h, d_k, attn_drop, proj_drop, bias)
        # Learnable residual-gating scalars alpha, beta in the paper.
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor,
                v_pad: Optional[torch.Tensor] = None,
                t_pad: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_v_new = h_v + self.alpha * self.v_from_t(h_v, h_t, key_padding_mask=t_pad)
        h_t_new = h_t + self.beta * self.t_from_v(h_t, h_v, key_padding_mask=v_pad)
        return h_v_new, h_t_new


class CDA(nn.Module):
    """Full Cross-Domain Attention module.

    Stack of `num_layers` CDA layers followed by LayerNorm and modality
    specific projection heads W_v, W_t (Eq. 12 in the manuscript).

    Pooling is "cls" (use the first token, matching CLIP / ViT convention)
    or "mean".
    """

    def __init__(self, d: int = 512, h: int = 8, d_k: Optional[int] = None,
                 num_layers: int = 1, pool: str = "cls",
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_proj_heads: bool = True,
                 bias: bool = True):
        super().__init__()
        assert pool in ("cls", "mean")
        self.pool = pool
        self.layers = nn.ModuleList([
            CDALayer(d=d, h=h, d_k=d_k, attn_drop=attn_drop,
                     proj_drop=proj_drop, bias=bias)
            for _ in range(num_layers)
        ])
        self.norm_v = nn.LayerNorm(d)
        self.norm_t = nn.LayerNorm(d)
        if use_proj_heads:
            self.W_v = nn.Linear(d, d, bias=bias)
            self.W_t = nn.Linear(d, d, bias=bias)
        else:
            self.W_v = None
            self.W_t = None

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor,
                v_pad: Optional[torch.Tensor] = None,
                t_pad: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_v, h_t = layer(h_v, h_t, v_pad=v_pad, t_pad=t_pad)
        h_v = self.norm_v(h_v)
        h_t = self.norm_t(h_t)
        if self.pool == "cls":
            v_g = h_v[:, 0]
            t_g = h_t[:, 0]
        else:
            v_g = h_v.mean(dim=1)
            t_g = h_t.mean(dim=1)
        if self.W_v is not None:
            v_g = self.W_v(v_g)
            t_g = self.W_t(t_g)
        return v_g, t_g


# ---------------------------------------------------------------------------
# Parameter-count utility.
# ---------------------------------------------------------------------------
def count_cda_parameters(d: int = 512, h: int = 8, d_k: int = 64,
                         num_layers: int = 1, use_proj_heads: bool = True,
                         bias: bool = True) -> dict:
    """Build the CDA module with the requested configuration and return a
    breakdown of its trainable parameters.

    The returned dict contains the total weight count and a few useful
    sub-totals so that the numbers reported in the paper can be checked.
    """
    model = CDA(d=d, h=h, d_k=d_k, num_layers=num_layers,
                use_proj_heads=use_proj_heads, bias=bias)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Closed-form expectation for the configuration with bias=True:
    #   per cross-attention block:  4 * (d*d + d)   (W_Q, W_K, W_V, W_O)
    #   per CDA layer            :  2 *  above  + 2 (alpha, beta)
    #   plus per CDA module      :  2 * (d*d + d)   (W_v, W_t projection heads)
    #                             +  4 * d          (two LayerNorms)
    per_block = 4 * (d * d + (d if bias else 0))
    per_layer = 2 * per_block + 2
    expected = num_layers * per_layer \
        + (2 * (d * d + (d if bias else 0)) if use_proj_heads else 0) \
        + 4 * d  # two LayerNorms (weight + bias each, dim = d)

    return {
        "d": d, "h": h, "d_k": d_k, "num_layers": num_layers,
        "use_proj_heads": use_proj_heads, "bias": bias,
        "per_block": per_block,
        "per_layer": per_layer,
        "expected_total": expected,
        "actual_total": total,
    }


if __name__ == "__main__":
    # Reproduce the configuration in the manuscript (h=8, d_k=64, d=512).
    print("=== Cross-Domain Attention parameter audit ===")
    print("(weights + biases for the bidirectional CDA module described in")
    print(" the manuscript, h=8, d_k=64, d=512, with LayerNorm and projection heads)\n")
    for L in (1, 2, 3, 6):
        info = count_cda_parameters(num_layers=L)
        print(f"  CDA layers = {L:>2d}  ->  trainable params = "
              f"{info['actual_total']:>12,d}  "
              f"(≈ {info['actual_total']/1e6:.2f} M)")
    print("\nBare (Q,K,V,O) accounting at the formula level:")
    print("  one direction (4 h d_k d) = 4 * 8 * 64 * 512 =",
          f"{4*8*64*512:,d} (= {4*8*64*512/1e6:.2f} M)")
    print("  one CDA layer (both directions) = 8 h d_k d =",
          f"{8*8*64*512:,d} (= {8*8*64*512/1e6:.2f} M)")
