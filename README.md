# Beyond Landmarks: Self-Supervised Pretraining for Label-Efficient Hand Gesture Recognition

## 📋 Project Resources

- **[Research Proposal](https://docs.google.com/document/d/11c9O-pnMULQ-t4CMpvxcZgKIL8rN2pk7wXHxR2nweKk/edit?tab=t.0)** - Detailed project proposal and methodology
- **[Project Notion](https://www.notion.so/CSC2503-Project-2a1f7375ce7e80029f22df1c2033ee28?source=copy_link)** - Project planning and progress tracking

## 🔬 Research Question

Can self-supervised pretraining on unlabeled hand video frames produce latent representations that outperform or complement landmark-based features for hand gesture classification, especially under low-label conditions?

## 💡 Motivation 

Hand gesture recognition is a fundamental component of human-computer interaction, with applications in VR/AR systems, accessibility, robotics, and remote collaboration. Traditional approaches fall into two camps:

**Landmark-based pipelines:** Extract 2D/3D keypoints and train sequential models on joint trajectories.
- ✅ **Pros:** compact, interpretable
- ❌ **Cons:** brittle to occlusion, errors propagate, discards appearance/context

**Raw video supervised learning:** Train CNNs/Transformers directly on RGB sequences.
- ✅ **Pros:** captures full visual information
- ❌ **Cons:** requires large annotated datasets

Both approaches are limited by the high annotation cost of gesture datasets. Recent advances in self-supervised learning: DINO, Masked Autoencoders, and VideoMAE have shown that deep models can learn strong visual representations from unlabeled data. 

This project explores whether self-supervised learning representations can enable label-efficient training, and serve as an effective alternative to landmarks for hand gesture recognition.

## 🔬 Methodology

- **Self-supervised Pretraining:** DINO / DINOv2 / DINOv3 / MAE / VideoMAE etc.
- **Fine-tuning & Classification:** Add a lightweight classification head. Fine-tune on small labeled gesture datasets. Compare label efficiency: train with 10%, 25%, and 100% of labels.
- **Baselines:** Landmark-based model: RNN/Transformer on MediaPipe keypoints. Supervised CNN/ViT. ImageNet-pretrained ViT.
- **Evaluation:** Gesture classification accuracy under different label fractions. Generalization under domain shifts. Embedding visualization.

---

# 📊 OUHANDS Dataset Analysis

## Overview
The OUHANDS (Oulu University HANDS) dataset is a comprehensive hand gesture recognition dataset containing multiple data modalities for hand detection and pose recognition.

## Original Paper
**"OUHANDS database for hand detection and pose recognition"** by M. Matilainen, P. Sangi, J. Holappa, et al. (2016)
- Published in 2016 Sixth International Conference on Image Processing Theory, Tools and Applications (IPTA)
- Cited by 77+ papers as of 2024

## Dataset Structure

### Main Directories
```
dataset/
├── OUHANDS_train/
│   ├── train/
│   │   ├── hand_data/         # Hand gesture images
│   │   ├── negative_data/     # Non-hand/background images
│   │   └── detection_supplemental_data/
│   ├── dlib/                  # Possibly face detection related
│   └── data_split_for_intermediate_tests/
├── OUHANDS_test/
│   └── test/
│       ├── hand_data/         # Test hand gesture images
│       └── negative_data/     # Test negative samples
├── annos/                     # XML annotations (PASCAL VOC format)
└── new_boxes/                 # Additional bounding box data
```

### Data Modalities
Each hand gesture has multiple representations:

#### Train/Test Hand Data:
- **colour/** - RGB color images (640x480 pixels)
- **depth/** - Depth information
- **bounding_box/** - Hand detection bounding boxes
- **orientation/** - Hand orientation data
- **segmentation/** - Hand segmentation masks

## File Naming Convention

### Hand Gesture Files
Pattern: `{CLASS}-{PERSON_ID}-{SAMPLE_ID}.png`

**Examples:**
- `A-ima-0001.png` → Class A, Person "ima", Sample 1
- `B-jha-0005.png` → Class B, Person "jha", Sample 5
- `K-vkb-0010.png` → Class K, Person "vkb", Sample 10

### Breakdown:
1. **Class Letter (A-K)**: Hand gesture/pose class
   - A, B, C, D, E, F, H, I, J, K (10 classes total)
   
2. **Person ID (3 characters)**: Unique identifier for each subject
   - Examples: ima, jha, jhb, jya, mka, mkb, mma, etc.
   - Appears to encode subject demographics or session info
   
3. **Sample Number (4 digits)**: Sequential sample number (0001-0010)
   - Each person typically has 10 samples per gesture class

### Negative Data Files
Pattern: `{NUMBER}.png`
- Examples: `0298.png`, `1804.png`, `1186.png`
- These are background/non-hand images for training detectors

## Annotation Format

XML files in PASCAL VOC format containing:
```xml
<annotation>
    <folder>hand_data</folder>
    <filename>A-ima-0001.png</filename>
    <source>
        <database>OUHANDS Database</database>
        <annotation>PASCAL VOC2007</annotation>
    </source>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>A</name>           <!-- Gesture class -->
        <pose>Right</pose>       <!-- Hand pose (Left/Right) -->
        <bndbox>
            <xmin>88</xmin>
            <ymin>78</ymin>
            <xmax>350</xmax>
            <ymax>360</ymax>
        </bndbox>
    </object>
</annotation>
```

## Dataset Statistics

- **Total Images**: 3,000 gesture images + ~17,026 negative samples
- **Training Images**: 1,600 (predefined split)
- **Validation Images**: 400 (predefined split)  
- **Test Images**: 1,000
- **Gesture Classes**: 10 (A, B, C, D, E, F, H, I, J, K)
- **Subjects**: ~20+ unique person IDs identified
- **Samples per person per class**: ~10
- **Image Resolution**: 640x480 pixels
- **Format**: PNG images with XML annotations

## Research Applications

Perfect for self-supervised pretraining research:

1. **Self-Supervised Pretraining**:
   - Use unlabeled RGB frames from `colour/` folders
   - Rich variety of hand poses and subjects
   - Multi-modal data (RGB + depth) available

2. **Label-Efficient Learning**:
   - Clear train/test splits provided
   - Can experiment with 10%, 25%, 100% of labels
   - Well-structured annotations for fine-tuning

3. **Baseline Comparisons**:
   - Bounding box data for detection baselines
   - Segmentation masks for detailed analysis
   - Compatible with MediaPipe for landmark extraction

## Key Features for This Project

- **Static Hand Gestures**: Perfect for image-based self-supervised learning
- **Multiple Subjects**: Good diversity for generalization
- **Multi-Modal**: RGB + depth data for comprehensive analysis
- **Well-Annotated**: PASCAL VOC format for easy integration
- **Reasonable Size**: Large enough for meaningful experiments, manageable for development

---

# 🤖 Hand Landmark Extraction

## Overview

The landmark extraction pipeline uses MediaPipe to extract 21 hand keypoints from all OUHANDS images, providing baseline features for comparison with self-supervised representations.

## Usage

Extract landmarks for all images in the dataset using OUHANDS predefined train/validation splits:

```bash
python landmark_extraction.py
```

### Command Line Options:

- `--dataset_path`: Path to OUHANDS dataset (default: `./dataset`)
- `--output_dir`: Output directory for landmarks (default: `./landmarks`)
- `--detection_confidence`: MediaPipe detection confidence (default: 0.7)

### Example with custom parameters:

```bash
python landmark_extraction.py \
    --dataset_path ./dataset \
    --output_dir ./landmarks \
    --detection_confidence 0.8
```

## Output Structure

After running, you'll have:

```
landmarks/
├── train/           # Training landmarks (1,600 files using OUHANDS predefined split)
├── validation/      # Validation landmarks (400 files using OUHANDS predefined split)
├── test/           # Test landmarks (~1,000 files)
└── extraction_report.json  # Processing statistics
```

Each landmark file contains:
- 21 hand landmarks (x, y, z coordinates)
- Handedness information (left/right hand)
- Original image metadata

## Landmark Format

Each `*_landmarks.json` file contains:

```json
{
  "image_path": "path/to/image.png",
  "image_shape": [480, 640, 3],
  "landmarks": [
    {"x": 0.5, "y": 0.3, "z": 0.1},
    ...  // 21 total landmarks
  ],
  "handedness": {
    "label": "Right",
    "score": 0.95
  }
}
```

## MediaPipe Hand Landmarks

The 21 landmarks follow MediaPipe's hand model:
- **0**: WRIST
- **1-4**: THUMB (CMC, MCP, IP, TIP)
- **5-8**: INDEX_FINGER (MCP, PIP, DIP, TIP)  
- **9-12**: MIDDLE_FINGER (MCP, PIP, DIP, TIP)
- **13-16**: RING_FINGER (MCP, PIP, DIP, TIP)
- **17-20**: PINKY (MCP, PIP, DIP, TIP)

## Usage in Research

1. **Baseline Model Training**: Use extracted landmarks to train RNN/Transformer models
2. **Comparison with Self-Supervised**: Compare landmark-based vs. raw image approaches
3. **Label Efficiency**: Experiment with different percentages of landmark annotations
4. **Multi-Modal Analysis**: Combine landmarks with raw images for hybrid approaches
