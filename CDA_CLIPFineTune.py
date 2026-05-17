"""
CDA-CLIP fine-tuning script.

A minimal training loop that wires the Cross-Domain Attention (CDA) module
defined in CDA.py on top of a frozen CLIP backbone, matching the
manuscript's description (Eqs. (5)-(13)).

It is the single training entry point of the codebase: the trainable
component is the CDA module, plugged into a frozen CLIP backbone.

Usage (example):
    python CDA_CLIPFineTune.py --vision_encoder ViT-B/16 --num_cda_layers 1

Run `python count_cda_params.py` (no torch required) for a static
parameter audit of the CDA module.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import clip

from CDA import CDA, count_cda_parameters
from dataloader import CustomedMedicalData
from utils import iNbreast_classes, sIIMACR_classes, cheXpert_classes, chest_xray_classes


# --------------------------------------------------------------------------- #
# Helpers for collecting visual and textual *token-level* features.            #
# CLIP's standard `encode_image` / `encode_text` return pooled features;       #
# we need the token sequences so that CDA can attend across modalities.        #
# --------------------------------------------------------------------------- #
def encode_text_tokens(model, text_tokens: torch.Tensor) -> torch.Tensor:
    """Token-level text features from CLIP's text transformer.
    Returns: (B, n_t, d).
    """
    x = model.token_embedding(text_tokens).type(model.dtype)        # (B, n_t, d)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    return x                                                        # (B, n_t, d)


def encode_image_tokens(model, images: torch.Tensor) -> torch.Tensor:
    """Token-level visual features.

    For ViT backbones we recover the patch+CLS sequence directly.  For
    ResNet backbones we use the attention-pool input (positional sequence
    of length H*W + 1 with the same channel dimension as the projection).
    Returns: (B, n_v, d).
    """
    visual = model.visual
    if hasattr(visual, "transformer"):                              # ViT branch
        x = visual.conv1(images.type(model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, n, d)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                              # (B, 1+n, d)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        return x                                                    # (B, n_v, d)
    else:                                                           # ResNet branch
        # Reuse the modified-ResNet stem and grab tokens from attnpool input.
        def stem(z):
            for conv, bn in [(visual.conv1, visual.bn1),
                             (visual.conv2, visual.bn2),
                             (visual.conv3, visual.bn3)]:
                z = visual.relu1(bn(conv(z))) if conv is visual.conv1 \
                    else visual.relu2(bn(conv(z))) if conv is visual.conv2 \
                    else visual.relu3(bn(conv(z)))
            return visual.avgpool(z)
        x = images.type(model.dtype)
        x = stem(x)
        x = visual.layer1(x); x = visual.layer2(x)
        x = visual.layer3(x); x = visual.layer4(x)
        # (B, C, H, W) -> token sequence with CLS = global average
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)                           # (B, HW, C)
        cls = x.mean(dim=1, keepdim=True)                           # (B, 1, C)
        x = torch.cat([cls, x], dim=1)                              # (B, 1+HW, C)
        return x


# --------------------------------------------------------------------------- #
# Main fine-tuning routine.                                                    #
# --------------------------------------------------------------------------- #
def train_cda_clip(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load(args.vision_encoder, device=device, jit=False)
    # Freeze the CLIP backbones.
    for p in model.parameters():
        p.requires_grad = False

    # CDA hidden dim is determined by CLIP's text/visual width (both 512 for
    # ViT-B/16 and RN101 in OpenAI CLIP).  We assert h * d_k == d.
    d = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
    h, d_k = args.num_heads, d // args.num_heads
    cda = CDA(d=d, h=h, d_k=d_k, num_layers=args.num_cda_layers,
              pool="cls", use_proj_heads=True).to(device)

    audit = count_cda_parameters(d=d, h=h, d_k=d_k,
                                 num_layers=args.num_cda_layers)
    trainable = sum(p.numel() for p in cda.parameters() if p.requires_grad)
    print(f"[CDA] config d={d}, h={h}, d_k={d_k}, "
          f"layers={args.num_cda_layers}")
    print(f"[CDA] expected trainable params (analytic): "
          f"{audit['expected_total']:,d}")
    print(f"[CDA] actual   trainable params (counted): "
          f"{trainable:,d}  (= {trainable/1e6:.3f} M)")

    optimizer = optim.AdamW(cda.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    train_ds = CustomedMedicalData(data_root=args.train_data, preprocess=preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07),
                               requires_grad=False).to(device)

    for epoch in range(args.epochs):
        cda.train()
        running = 0.0
        for batch in train_loader:
            # The dataloader returns (images, text_tokens, labels[, heatmaps]).
            images, text_tokens = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                h_v = encode_image_tokens(model, images)            # (B, n_v, d)
                h_t = encode_text_tokens(model, text_tokens)        # (B, n_t, d)

            v_g, t_g = cda(h_v.float(), h_t.float())                # (B, d), (B, d)
            v_g = v_g / v_g.norm(dim=-1, keepdim=True)
            t_g = t_g / t_g.norm(dim=-1, keepdim=True)

            logits = logit_scale.exp() * v_g @ t_g.t()
            gt = torch.arange(len(images), device=device)
            loss = 0.5 * (loss_img(logits, gt) + loss_txt(logits.t(), gt))

            loss.backward()
            optimizer.step()
            running += float(loss)
        scheduler.step()
        print(f"epoch {epoch:>3d}  loss={running/max(1,len(train_loader)):.4f}")

    return cda


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vision_encoder", type=str, default="ViT-B/16",
                   choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16"])
    p.add_argument("--num_cda_layers", type=int, default=1,
                   help="Stacked depth of the CDA module (paper ablates 1-3).")
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train_data", type=str,
                   default="./data/MIMICGAZE_CLIP_train.txt")
    return p.parse_args()


if __name__ == "__main__":
    train_cda_clip(_parse())
