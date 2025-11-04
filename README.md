# Beyond Landmarks: Self-Supervised Pretraining for Label-Efficient Hand Gesture Recognition

## Research Question

Can self-supervised pretraining on unlabeled hand video frames produce latent representations that outperform or complement landmark-based features for hand gesture classification, especially under low-label conditions?

## Motivation 

Hand gesture recognition is a fundamental component of human-computer interaction, with applications in VR/AR systems, accessibility, robotics, and remote collaboration. Traditional approaches fall into two camps:
Landmark-based pipelines: Extract 2D/3D keypoints and train sequential models on joint trajectories.
- Pros: compact, interpretable
- Cons: brittle to occlusion, errors propagate, discards appearance/context.
Raw video supervised learning: Train CNNs/Transformers directly on RGB sequences.
- Pros: captures full visual information
- Cons: requires large annotated datasets
Both approaches are limited by the high annotation cost of gesture datasets. Recent advances in self-supervised learning: DINO, Masked Autoencoders, and VideoMAE have shown that deep models can learn strong visual representations from unlabeled data. 
This project explores whether self-supervised learning representations can enable label-efficient training, and serve as an effective alternative to landmarks for hand gesture recognition. 

## Methodology

- Self-supervised Pretraining: DINO / DINOv2 / DINOv3 / MAE / VideoMAE etc.
- Fine-tuning & Classification: Add a lightweight classification head. Fine-tune on small labeled gesture datasets. Compare label efficiency: train with 10%, 25%, and 100% of labels.
- Baselines: Landmark-based model: RNN/Transformer on MediaPipe keypoints. Supervised CNN/ViT. ImageNet-pretrained ViT.
- Evaluation: Gesture classification accuracy under different label fractions. Generalization under domain shifts. Embedding visualization.
