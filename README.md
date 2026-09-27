# CDA-CLIP

This repository contains the implementation of CDA-CLIP for the manuscript:

**Latent Representation Alignment for Medical Multimodal Learning**

## Overview

CDA-CLIP is a CLIP-based medical multimodal learning framework that introduces a Cross-Domain Attention (CDA) module to improve image-text alignment for medical imaging tasks.

## Files

- `CDA.py`: implementation of the Cross-Domain Attention module.
- `CDA_CLIPFineTune.py`: training or fine-tuning script for CDA-CLIP.
- `count_cda_params.py`: script for counting CDA-related parameters.
- `dataloader.py`: data loading utilities.
- `GradCAM_CDA_CLIP.py`: Grad-CAM visualization script.
- `parameters.py`: parameter settings.
- `PrepareDatasets.py`: dataset preparation script.
- `utils.py`: utility functions.

## Manuscript

Latent Representation Alignment for Medical Multimodal Learning

## Authors

Dongni Deng, Baoying Yu, and Ziwei Chen.

## Code Availability

The source code is provided for research and reproducibility purposes.

---

## Reproducibility notes for the 2026 revision

The current revision uses a frozen CLIP backbone and trains only the
Cross-Domain Attention (CDA) module and modality-specific projection heads.

### Main CDA-CLIP configuration

- CLIP backbone: ViT-B/16
- CDA layers: 1
- Attention heads: 8
- Optimizer: AdamW
- Learning rate: 2e-4
- Weight decay: 5
- Batch size: 16
- Epochs: 50
- Random seed: 42
- Contrastive temperature: 0.07 (fixed)
- Text pooling: CLIP end-of-text (EOT) token
- Hyperparameter selection: patient-disjoint source-domain cross-validation
- External target datasets are not used for hyperparameter selection

### Main scripts

`CDA_CLIPFineTune_CV_EOT.py`
Source-domain cross-validation with frozen CLIP and EOT text pooling.

`train_original_eot_eval_all4.py`
Final ViT-B/16 CDA-CLIP training/evaluation pathway.

`train_original_eot_eval_all4_rn101.py`
Matched RN101 CDA-CLIP training pathway.

`eval_eot_fullscope_shared.py`
Unified full-scope external evaluation for the ViT-B/16 setting.

`eval_rn101_fullscope_clean.py`
Matched RN101 full-scope evaluation and prompt-ensemble evaluation.

`domain_gap_quantitative.py`
Quantitative natural-to-medical feature-space analysis using frozen CLIP
features, MMD, and a domain-classification probe.

`domain_gap_verify.py`
Independent NumPy-based verification of the domain-gap analysis.

### CDA-MedCLIP transfer

`train_cda_medclip_gamma_cv.py`
Source-domain gamma cross-validation.

`rank_cda_medclip_gamma_pairwise_val.py`
True-pairwise source-validation ranking used for gamma selection.

`train_cda_medclip_gamma_full.py`
Final CDA-MedCLIP training for the residual-strength analysis.

`eval_cda_medclip_gamma_full.py`
Gamma sensitivity evaluation.

`eval_cda_medclip_gamma_fullscope.py`
Unified full-scope CDA-MedCLIP evaluation.

### Baseline reproduction

The `baseline_suite/` directory contains the scripts used for the
standardized full-scope baseline evaluations and prompt-compatible
comparisons.

The public medical datasets are not redistributed in this repository.
Users should download them from their original sources. Dataset locations
can be configured locally. For the baseline scripts, the project root can
also be supplied with the environment variable:

    export CDA_CLIP_PROJECT_ROOT=/path/to/CDA-CLIP

Model checkpoints and raw experimental outputs are intentionally excluded
from version control.

