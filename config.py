"""Configuration objects for DINO fine-tuning experiments.

All hyperparameters and experiment level settings live here so that
training and evaluation scripts can share a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """Settings related to dataset loading."""

    root_dir: str = "./dataset"
    batch_size: int = 32
    num_workers: int = 4
    train_subset_ratio: float = 1.0
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model specific settings."""

    model_name: str = "vit_small_patch16_dinov3"
    num_classes: int = 10
    unfreeze_blocks: int = 3  # Number of final transformer blocks to fine-tune
    swav_arch: str = "swav_resnet50"
    swav_trainable: bool = True
    attention_threshold: float = 0.6
    attention_margin: float = 0.15
    fusion_dim: int | None = None


@dataclass
class TrainingConfig:
    """Optimizer and training loop settings."""

    epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    scheduler_t_max: int = 5
    segmentation_kl_weight: float = 0.1


@dataclass
class ExperimentConfig:
    """Top level experiment settings."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("./checkpoints")
    run_name: str = "vit_small_patch16_dinov3_k4"

    def checkpoint_path(self, mode='best') -> Path:
        """Return the path for saving/loading experiment checkpoints."""
        return Path(self.output_dir) / f"{self.run_name}_{mode}.pt"


def get_config() -> ExperimentConfig:
    """Return a default experiment configuration instance."""
    return ExperimentConfig()
