"""Joint image/mask transforms to keep segmentation aligned with augmentations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass
class SegmentationTrainTransform:
    """Apply train-time augmentations consistently to images and masks."""

    size: int
    mean: Iterable[float]
    std: Iterable[float]
    scale: Tuple[float, float] = (0.67, 1.0)
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip_prob: float = 0.5
    color_jitter_strength: float = 0.4

    def __post_init__(self) -> None:
        self.interpolation = InterpolationMode.BICUBIC
        strength = self.color_jitter_strength
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=strength,
                contrast=strength,
                saturation=strength,
                hue=strength / 4.0,
            )
            if strength > 0
            else None
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(image, Image.Image):
            image = TF.to_pil_image(image)
        if mask is not None and not isinstance(mask, Image.Image):
            mask = TF.to_pil_image(mask)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )
        image = TF.resized_crop(image, i, j, h, w, (self.size, self.size), self.interpolation)
        if mask is not None:
            mask = TF.resized_crop(
                mask, i, j, h, w, (self.size, self.size), InterpolationMode.NEAREST
            )

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)

        mask_tensor: Optional[torch.Tensor] = None
        if mask is not None:
            mask_array = np.array(mask, dtype=np.float32)
            mask_array = mask_array / 255.0
            mask_tensor = torch.from_numpy(mask_array)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            else:
                mask_tensor = mask_tensor[:1, ...]

        return image_tensor, mask_tensor


@dataclass
class SegmentationEvalTransform:
    """Deterministic transforms for validation/test with segmentation masks."""

    size: int
    mean: Iterable[float]
    std: Iterable[float]
    resize_factor: float = 1.14  # roughly 256 for 224 inputs

    def __post_init__(self) -> None:
        self.resize_size = int(round(self.size * self.resize_factor))
        self.interpolation = InterpolationMode.BICUBIC

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(image, Image.Image):
            image = TF.to_pil_image(image)
        if mask is not None and not isinstance(mask, Image.Image):
            mask = TF.to_pil_image(mask)

        image = TF.resize(image, (self.resize_size, self.resize_size), self.interpolation)
        image = TF.center_crop(image, (self.size, self.size))

        if mask is not None:
            mask = TF.resize(mask, (self.resize_size, self.resize_size), InterpolationMode.NEAREST)
            mask = TF.center_crop(mask, (self.size, self.size))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)

        mask_tensor: Optional[torch.Tensor] = None
        if mask is not None:
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            else:
                mask_tensor = mask_tensor[:1, ...]

        return image_tensor, mask_tensor
