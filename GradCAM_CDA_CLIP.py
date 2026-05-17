"""
Grad-CAM visualisation for CDA-CLIP.

This script reproduces the visualisation used in the manuscript's
qualitative comparison between vanilla CLIP and CDA-CLIP.  It hooks the
last visual feature map of the (frozen) CLIP backbone and back-propagates
the cosine similarity between the *CDA-enhanced* image embedding and the
*CDA-enhanced* text embedding produced by the trained CDA module.

Required artefacts:
  - CDA.py            : the Cross-Domain Attention module.
  - CDA_CLIPFineTune.py : provides ``encode_image_tokens`` and
                          ``encode_text_tokens`` to obtain the token-level
                          CLIP features that CDA attends over.

Visualises the CDA-augmented attention against the vanilla CLIP attention
for the same image and caption.
"""

import argparse
import os
import urllib.request
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import filters
from torch import nn

import clip
import matplotlib.pyplot as plt

from CDA import CDA
from CDA_CLIPFineTune import encode_image_tokens, encode_text_tokens


# --------------------------------------------------------------------------- #
# Visualisation helpers                                                       #
# --------------------------------------------------------------------------- #
def normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img: np.ndarray, attn_map: np.ndarray, blur: bool = True) -> np.ndarray:
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap("jet")
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    return (
        (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img
        + (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    )


def viz_attn(img: np.ndarray, attn_cda: np.ndarray, attn_clip: np.ndarray,
             out_path: str = None, blur: bool = True) -> None:
    _, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(img);           axes[0].set_title("input")
    axes[1].imshow(getAttMap(img, attn_clip, blur)); axes[1].set_title("CLIP")
    axes[2].imshow(getAttMap(img, attn_cda, blur));  axes[2].set_title("CDA-CLIP")
    for ax in axes:
        ax.axis("off")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def load_image(img_path: str, resize: int = None) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Grad-CAM core                                                               #
# --------------------------------------------------------------------------- #
class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self._save)

    def _save(self, module, _input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self): return self
    def __exit__(self, *exc): self.hook.remove()

    @property
    def activation(self) -> torch.Tensor: return self.data
    @property
    def gradient(self) -> torch.Tensor: return self.data.grad


def gradCAM_against_target(forward_fn, target: torch.Tensor,
                           layer: nn.Module) -> torch.Tensor:
    """Generic Grad-CAM: backprop ``forward_fn()`` onto ``target`` via hook
    on ``layer`` and return a (1, 1, H, W) saliency map.
    ``forward_fn`` must compute a scalar-compatible tensor of the same
    shape as ``target`` (the dot/cosine product is taken outside).
    """
    with Hook(layer) as hook:
        out = forward_fn()                    # (B, d)
        score = (out * target).sum()
        score.backward()
        grad = hook.gradient.float()
        act = hook.activation.float()
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(act * alpha, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
    return cam


# --------------------------------------------------------------------------- #
# End-to-end CDA-CLIP saliency                                                #
# --------------------------------------------------------------------------- #
def cda_clip_saliency(model, cda: CDA, image_input: torch.Tensor,
                      text_tokens: torch.Tensor, saliency_layer: str,
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (attn_map_cda, attn_map_clip) at the resolution of the input."""
    # ------------------------------------------------------------------ CLIP
    layer = getattr(model.visual, saliency_layer)

    def forward_clip():
        return model.visual(image_input).float()

    text_emb_clip = model.encode_text(text_tokens).float()
    text_emb_clip = text_emb_clip / text_emb_clip.norm(dim=-1, keepdim=True)
    cam_clip = gradCAM_against_target(forward_clip, text_emb_clip, layer)

    # ------------------------------------------------------------------ CDA
    def forward_cda():
        h_v = encode_image_tokens(model, image_input).float()
        h_t = encode_text_tokens(model, text_tokens).float()
        v_g, _ = cda(h_v, h_t)
        v_g = v_g / v_g.norm(dim=-1, keepdim=True)
        return v_g

    # Text side of CDA is computed once for the saliency target.
    with torch.no_grad():
        h_v0 = encode_image_tokens(model, image_input).float()
        h_t0 = encode_text_tokens(model, text_tokens).float()
        _, t_g = cda(h_v0, h_t0)
        t_g = t_g / t_g.norm(dim=-1, keepdim=True)

    cam_cda = gradCAM_against_target(forward_cda, t_g, layer)

    # Resize to input resolution.
    H, W = image_input.shape[-2:]
    cam_cda = F.interpolate(cam_cda, (H, W), mode="bicubic", align_corners=False)
    cam_clip = F.interpolate(cam_clip, (H, W), mode="bicubic", align_corners=False)
    return (cam_cda.squeeze().detach().cpu().numpy(),
            cam_clip.squeeze().detach().cpu().numpy())


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vision_encoder", type=str, default="RN101",
                   choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16"])
    p.add_argument("--saliency_layer", type=str, default="layer4")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--caption", type=str, default="collapsed lung")
    p.add_argument("--cda_ckpt", type=str, default=None,
                   help="Path to a trained CDA checkpoint (state_dict).")
    p.add_argument("--num_cda_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--out", type=str, default=None,
                   help="Optional output figure path.")
    return p.parse_args()


def main():
    args = _parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load(args.vision_encoder, device=device, jit=False)
    for p in model.parameters():
        p.requires_grad = False

    d = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
    cda = CDA(d=d, h=args.num_heads, d_k=d // args.num_heads,
              num_layers=args.num_cda_layers).to(device)
    if args.cda_ckpt and os.path.isfile(args.cda_ckpt):
        cda.load_state_dict(torch.load(args.cda_ckpt, map_location=device))
        print(f"loaded CDA weights from {args.cda_ckpt}")

    image_input = preprocess(Image.open(args.image_path)).unsqueeze(0).to(device)
    image_np = load_image(args.image_path, model.visual.input_resolution)
    text_tokens = clip.tokenize([args.caption]).to(device)

    attn_cda, attn_clip = cda_clip_saliency(
        model, cda, image_input, text_tokens, args.saliency_layer, device
    )
    viz_attn(image_np, attn_cda=attn_cda, attn_clip=attn_clip, out_path=args.out)


if __name__ == "__main__":
    main()
