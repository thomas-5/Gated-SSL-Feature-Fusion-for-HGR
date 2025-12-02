# Gated Self-Supervised Feature Fusion for Label-Efficient Hand Gesture Recognition

This repository explores a label-efficient approach that combines spatially structured ViT features learned by self-supervised methods with instance-discriminative CNN features. The core idea is a two-stream fusion architecture that leverages attention from a ViT to localize the hand and guide a second stream that extracts complementary features.Our work investigates whether self-supervised learning features can provide label-efficient hand gesture recognition on OUHANDS. The repository ships production-ready pipelines for our proposed model and visualization tooling. You can find the report paper [here](report.pdf).

## System Highlights

- DINO ViT fine-tuning with optional segmentation-guided attention KL loss and mixed precision support.
- SwAV fusion head that crops attention-derived hand regions, extracts SwAV features, and fuses them with DINO CLS tokens through a learned gate.
- Supervised ImageNet baseline leveraging `vit_base_patch16_224` with selective layer unfreezing.
- Shared dataloading stack, deterministic seeding, and paired transforms for aligned image–mask augmentations.
- Automated leakage diagnostics across train, validation, and test splits.
- Attention map export plus t-SNE embedding inspection for any checkpoint.

## Codebase Overview

- `train.py`: model training loop with cosine schedule, AMP, segmentation-aware regularization, and checkpoint management.
- `evaluate.py`: validation/test runner that reports accuracy, precision, recall, F1, and saves attention versus segmentation grids.
- `checkpoints/`: store model checkpoints,
- `model.py`: proposed model construction, partial unfreezing, and transform provisioning (including paired segmentation transforms).
- `utils.py`: device/seed helpers, forward-with-attention wrapper, deterministic sampling utilities.
- `paired_transforms.py`: torchvision-style transforms that keep image/mask pairs spatially aligned.
- `ouhands_loader.py`: dataset implementation supporting RGB images, segmentation masks, and paired transforms.
- `config.py` : experiment-level hyperparameters and dataset settings.

## Methodology

- Two-stream architecture:

  - Stream A (DINO / ViT): extracts spatial ViT features and produces a global CLS embedding. The ViT attention maps are used to localize hand regions.
  - Stream B (SwAV / ResNet): processes an attention-guided crop of the input image to produce instance-discriminative features focused on the hand region.
- Attention-Guided Cropping:

  - Compute a hand heatmap from the ViT's last-block CLS→patch attention.
  - Derive a minimum bounding rectangle around high-attention patches, optionally expand by a small margin, and crop the input image.
  - Feed the crop to the SwAV encoder to obtain a complementary feature vector.
- Gated Feature Fusion:

  - Project each stream's features to a common embedding dimension.
  - Compute a learned gate from the concatenated stream features and apply a sigmoid to obtain per-dimension weights.
  - Fuse projected stream features with the gate as a convex combination and pass the fused vector to a classifier head.
- Segmentation-Guided Attention Regularization:

  - When segmentation masks are available during training, apply a KL loss that encourages the ViT attention distribution to align with the ground-truth hand mask.
  - Final training objective: L_final = L_CE + lambda * L_SG, where L_CE is cross-entropy on class labels and L_SG is the attention KL term.

The network is trained end-to-end for classification while the gated fusion and segmentation guidance improve localization and robustness with relatively few labeled examples.

## Dataset overview

- Download dataset: [OUHANDS](https://www.kaggle.com/datasets/mumuheu/ouhands) (static hand gesture classes).
- Put the downloaded dataset under the `dataset/` folder.
- Standard train / validation / test splits are used; see the dataset folders for exact counts.

## How to use

- Train the GSSL feature fusion model:

  ```bash
  python train.py
  ```
- Evaluate a checkpoint:

  ```bash
  python evaluate.py --checkpoint-path checkpoints/<checkpoint>.pt
  ```

You can download the best checkpoint [here](https://drive.google.com/file/d/1vN2fS9ulhDepRnbtbp1yh9wtsAUsOJj1/view?usp=sharing), and save it to `checkpoints/vit_small_patch16_dinov3_k4_best.pt`

```txt
accuracy=0.9510 
precision=0.9544 
recall=0.9510 
f1=0.9515
```

By default: `python evaluate.py` will evaluate the this model, and `python train.py` will overwrite this model.

For additional configuration (hyperparameters, segmentation loss weight, grid-search options), see `config.py` and the top of each script.
