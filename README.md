# Beyond Landmarks: Self-Supervised Pretraining for Label-Efficient Hand Gesture Recognition

Beyond Landmarks investigates whether self-supervised Vision Transformers can provide label-efficient hand gesture recognition on OUHANDS. The repository ships production-ready pipelines for DINO fine-tuning, a supervised ViT baseline, diagnostics, and visualization tooling.

## Project Resources

- Research proposal: https://docs.google.com/document/d/11c9O-pnMULQ-t4CMpvxcZgKIL8rN2pk7wXHxR2nweKk/edit?tab=t.0
- Project space: https://www.notion.so/CSC2503-Project-2a1f7375ce7e80029f22df1c2033ee28?source=copy_link

## System Highlights

- DINO ViT fine-tuning with optional segmentation-guided attention KL loss and mixed precision support.
- Optional SwAV fusion head that crops attention-derived hand regions, extracts SwAV features, and fuses them with DINO CLS tokens through a learned gate.
- Supervised ImageNet baseline leveraging `vit_base_patch16_224` with selective layer unfreezing.
- Shared dataloading stack, deterministic seeding, and paired transforms for aligned image–mask augmentations.
- Automated leakage diagnostics across train, validation, and test splits.
- Attention map export plus t-SNE embedding inspection for any checkpoint.

## Codebase Overview

- `train.py`: DINO fine-tuning loop with cosine schedule, AMP, segmentation-aware regularization, and checkpoint management.
- `evaluate.py`: validation/test runner that reports accuracy, precision, recall, F1, and saves attention versus segmentation grids.
- `model.py`: timm ViT construction, partial unfreezing, and transform provisioning (including paired segmentation transforms).
- `utils.py`: device/seed helpers, forward-with-attention wrapper, deterministic sampling utilities.
- `paired_transforms.py`: torchvision-style transforms that keep image/mask pairs spatially aligned.
- `train_supervised_vit.py`: supervised baseline training script using the shared loaders and GradScaler-aware optimization.
- `check_data_leakage.py`: split verification via filename, path, and hash comparisons.
- `tsne_dino.py` and `tsne_supervised.py`: single-checkpoint t-SNE embedding visualizers for DINO and supervised models.
- `ouhands_loader.py`: dataset implementation supporting RGB images, bounding boxes, segmentation masks, and paired transforms.
- `config.py` and `landmark_configs.yaml`: experiment-level hyperparameters and dataset settings.

## Setup

1. Create a Python 3.10+ environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Verify CUDA drivers are available for GPU execution; AMP activates automatically on CUDA devices.

## Data Preparation

1. Place the OUHANDS dataset under `dataset/` following the directory layout expected by `ouhands_loader.py`.
2. Optional: run `python check_data_leakage.py --root-dir dataset` to confirm split integrity before training.
3. Segmentation masks are required only when using attention regularization; bounding boxes and RGB images are always supported.

## Training Pipelines

### DINO Fine-Tuning

`python train.py`

- Pulls configuration from `config.py` (output paths, augmentation choices, optimizer settings).
- Supports segmentation-weighted KL loss via `config.training.segmentation_kl_weight`.
- Enable SwAV fusion by setting `config.model.use_swav_fusion = True`; the script will derive hand-centric crops from attention maps and fuse SwAV/DINO embeddings before classification.
- Saves `*_best.pt` and `*_last.pt` checkpoints under `checkpoints/`.

### Supervised ViT Baseline

`python train_supervised_vit.py`

- Loads an ImageNet-pretrained `vit_base_patch16_224`, freezes early blocks, and fine-tunes the last `k` transformer blocks plus the classification head.
- Shares dataloaders and evaluation routines with the DINO pipeline.
- Outputs `vit_supervised_best.pt` in `checkpoints/`.

## Evaluation

`python evaluate.py --checkpoint-path checkpoints/<file>.pt`

- Reports validation loss/accuracy and test accuracy, precision, recall, and macro F1.
- Generates attention-versus-segmentation grids under `outputs/` when segmentation masks are available.
- Uses `utils.forward_with_attention` to recover attention maps for qualitative inspection.

## Visualization Tools

- Attention maps: `evaluate.save_attention_grid` samples one image per class (deterministic seed), overlays attention heatmaps, and optionally renders segmentation masks in a 3×8 grid.
- t-SNE embeddings: run `python tsne_dino.py checkpoints/<dino_checkpoint>.pt --output outputs/tsne_dino.png` for DINO checkpoints or `python tsne_supervised.py checkpoints/<supervised_checkpoint>.pt --arch vit_base_patch16_224 --output outputs/tsne_supervised.png` for supervised checkpoints.

Both entry points reuse `create_eval_dataloaders`, ensuring embeddings are drawn from identical preprocessing pipelines.

## Diagnostics

`python check_data_leakage.py --root-dir dataset`

- Flags any shared samples across splits by comparing canonicalized paths and SHA256 hashes.
- Prints actionable reports highlighting duplicates, if any.

## OUHANDS Dataset Summary

- 3,000 gesture images plus roughly 17,000 negatives distributed across OUHANDS train, validation, and test folders.
- Ten static gesture classes (A, B, C, D, E, F, H, I, J, K) with approximately ten samples per subject per class.
- Multimodal assets include RGB, depth, segmentation masks, bounding boxes, and orientation metadata; this project uses RGB and optional masks.
- Annotations follow the PASCAL VOC XML schema; images are 640×480 PNGs.

Refer to `dataset/annos/` for annotation examples and `__MACOSX/` for legacy distribution artifacts.

## References

- Matilainen, M. et al. "OUHANDS database for hand detection and pose recognition", IPTA 2016.
- Caron, M. et al. "Emerging Properties in Self-Supervised Vision Transformers", 2021.
- Dosovitskiy, A. et al. "An Image is Worth 16x16 Words", 2020.

