# Learned Landmarks: MobileNet for Hand Landmark Regression

This folder contains a PyTorch implementation for training a MobileNetV2 model to directly regress hand landmarks from images, using MediaPipe-extracted landmarks as ground truth.

## Files

- **`landmarks_loader.py`**: Dataset class for loading OUHANDS images and corresponding MediaPipe landmarks
- **`mobilenet_v2.py`**: MobileNetV2-based landmark regression model
- **`training.py`**: Main training script
- **`training_utils.py`**: Training utilities (train_one_epoch, evaluate, MPJPE metric)
- **`test_training.py`**: Test script for validating the implementation with a small subset

## Dataset Structure

The implementation expects:
- Images from OUHANDS dataset in `dataset/OUHANDS_train/train/hand_data/colour/`
- MediaPipe landmarks in `landmarks/{train,validation,test}/` folders
- Each landmark file contains 21 3D hand landmarks (63 values total)

## Usage

### Quick Test
```bash
cd models/learned_landmarks
python test_training.py
```

### Full Training
```bash
cd models/learned_landmarks
python training.py
```

## Model Architecture

- **Backbone**: MobileNetV2 (from torchvision)
- **Head**: Dropout → Linear(1280→512) → ReLU → Dropout → Linear(512→63)
- **Output**: 63-dimensional vector (21 landmarks × 3 coordinates each)
- **Loss**: SmoothL1Loss with β=0.02
- **Metric**: Mean Per-Joint Position Error (MPJPE)

## Key Features

- **Data Loading**: Automatically loads train/validation splits with hand landmark filtering
- **Preprocessing**: 
  - Images: Resize to 224×224, ImageNet normalization
  - Landmarks: Optional standardization to [-1,1] range
- **Augmentation**: ColorJitter and RandomGrayscale for training set
- **Training**: Mixed precision, cosine LR scheduling, model checkpointing
- **Evaluation**: MPJPE metric for landmark accuracy

## Configuration Options

### Dataset Parameters
- `split`: 'train', 'validation', 'test'
- `image_size`: Input image size (default: 224)
- `augment`: Enable data augmentation (default: True for train)
- `standardize`: Normalize landmark coordinates (default: True)

### Training Parameters
- `learning_rate`: 3e-3
- `batch_size`: 64 (GPU) / 16 (CPU)
- `epochs`: 50
- `weight_decay`: 1e-4

## Results from Test Run

On a small subset (32 train, 8 validation samples):
- **Epoch 1**: train_mpjpe=0.800, val_mpjpe=0.683
- **Epoch 3**: train_mpjpe=0.516, val_mpjpe=0.470

The model successfully learns to predict hand landmarks from images, with MPJPE decreasing consistently during training.

## Note for Colab Usage

The implementation is designed to work seamlessly in Google Colab:
- No argparse or config file dependencies
- Automatic GPU/CPU detection
- Simple import structure
- Relative path handling for dataset access