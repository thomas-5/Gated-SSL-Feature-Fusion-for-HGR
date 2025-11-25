import os
import sys
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class OuhandsDS(Dataset):
    """
    PyTorch Dataset for OUHANDS hand gesture recognition.
    
    Provides raw images with gesture class labels and optional bounding box annotations.
    Suitable for training classification models, pose estimation models, or any other 
    computer vision tasks on hand gestures.
    
    Args:
        root_dir (str, optional): Root directory containing OUHANDS data. 
                                 Defaults to './dataset'
        split (str): Dataset split - 'train', 'validation', or 'test'
        transform (callable, optional): Optional transform to be applied to images
        target_transform (callable, optional): Optional transform for labels
        return_paths (bool): If True, return path as additional output
        class_subset (list, optional): Only include specific gesture classes (e.g., ['A', 'B', 'C'])
        use_bounding_box (bool): If True, load and return bounding box information
        crop_to_bbox (bool): If True, crop images to bounding box region (requires use_bounding_box=True)
        bbox_padding (int): Padding pixels around bounding box when cropping
        train_subset_ratio (float): Fraction of training data to use (0.0 to 1.0). Only applies to 'train' split.
                                   Useful for label efficiency experiments. Default is 1.0 (use all data).
        random_seed (int): Random seed for reproducible subset sampling. Default is 42.
    
    Attributes:
        classes (list): List of gesture class names ['A', 'B', ..., 'K']
        class_to_idx (dict): Mapping from class name to integer index
        samples (list): List of (image_path, class_index) tuples
        split (str): Current dataset split
        use_bounding_box (bool): Whether bounding box data is loaded
        crop_to_bbox (bool): Whether images are cropped to bounding box
    """
    
    # OUHANDS gesture classes (10 classes: A-K excluding G)
    GESTURE_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K']

    def __init__(
        self,
        root_dir: str = r'D:\Courses\Csc2503\proj\archive',
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_paths: bool = False,
        class_subset: Optional[List[str]] = None,
        use_bounding_box: bool = False,
        crop_to_bbox: bool = False,
        bbox_padding: int = 20,
        train_subset_ratio: float = 1.0,
        random_seed: int = 42
    ):

        # Validate arguments
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Split must be 'train', 'validation', or 'test', got '{split}'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.use_bounding_box = use_bounding_box
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding
        self.train_subset_ratio = train_subset_ratio
        self.random_seed = random_seed

        # Validate train_subset_ratio
        if not 0.0 <= train_subset_ratio <= 1.0:
            raise ValueError(f"train_subset_ratio must be between 0.0 and 1.0, got {train_subset_ratio}")

        # Handle class subset
        if class_subset is not None:
            invalid_classes = set(class_subset) - set(self.GESTURE_CLASSES)
            if invalid_classes:
                raise ValueError(f"Invalid classes: {invalid_classes}. Valid classes: {self.GESTURE_CLASSES}")
            self.classes = sorted(class_subset)
        else:
            self.classes = self.GESTURE_CLASSES.copy()

        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load dataset samples
        self.samples = self._load_samples()

        # Create default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])


    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image samples and their corresponding class labels."""

        samples = []
        class_counts = {cls: 0 for cls in self.classes}

        if self.split in ['train', 'validation']:
            # Use proper train-validation splits from data_split_for_intermediate_tests
            split_file_mapping = {
                'train': 'OUHANDS_train/data_split_for_intermediate_tests/training_files.txt',
                'validation': 'OUHANDS_train/data_split_for_intermediate_tests/validation_files.txt'
            }
            
            split_file_path = self.root_dir / split_file_mapping[self.split]
            base_image_dir = self.root_dir / 'OUHANDS_train/train/hand_data/colour'
            
            if not split_file_path.exists():
                raise RuntimeError(f"Split file not found: {split_file_path}")
            
            # Read filenames from split file
            with open(split_file_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
            
            # Load samples based on split file
            for filename in filenames:
                image_path = base_image_dir / filename
                
                if not image_path.exists():
                    print(f"Warning: Image file not found: {image_path}")
                    continue
                
                # Extract gesture class from filename
                try:
                    gesture_class = filename.split('-')[0].upper()
                except IndexError:
                    print(f"Warning: Skipping file with unexpected name format: {filename}")
                    continue

                # Check if this class is in our subset
                if gesture_class not in self.classes:
                    continue

                # Get class index
                class_idx = self.class_to_idx[gesture_class]

                samples.append((image_path, class_idx))
                class_counts[gesture_class] += 1
                
        else:  # test split
            # Use test directory for test split only
            split_dir = self.root_dir / 'OUHANDS_test/test/hand_data/colour'
            
            # Supported image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

            # Find all image files
            for image_path in split_dir.iterdir():
                if not image_path.is_file():
                    continue

                # Check if it's an image file
                if image_path.suffix.lower() not in image_extensions:
                    continue

                # Extract gesture class from filename
                # Expected format: <CLASS>-<subject>-<number>.<ext> (e.g., A-ima-0001.png, B-jha-0002.png)
                filename = image_path.stem

                try:
                    gesture_class = filename.split('-')[0].upper()
                except IndexError:
                    print(f"Warning: Skipping file with unexpected name format: {image_path.name}")
                    continue

                # Check if this class is in our subset
                if gesture_class not in self.classes:
                    continue

                # Get class index
                class_idx = self.class_to_idx[gesture_class]

                samples.append((image_path, class_idx))
                class_counts[gesture_class] += 1

        if not samples:
            raise RuntimeError(f"No valid samples found for split '{self.split}'")

        # Sort samples by filename for reproducible ordering
        samples.sort(key=lambda x: x[0].name)

        # Apply subset sampling for training data if train_subset_ratio < 1.0
        if self.split == 'train' and self.train_subset_ratio < 1.0:
            original_count = len(samples)
            samples = self._sample_subset(samples, class_counts)
            print(f"Sampled {len(samples)}/{original_count} samples ({self.train_subset_ratio*100:.1f}%) for {self.split} split")
        else:
            print(f"Loaded {len(samples)} samples for {self.split} split")

        # Recompute class counts after sampling
        final_class_counts = {cls: 0 for cls in self.classes}
        for _, class_idx in samples:
            class_name = self.classes[class_idx]
            final_class_counts[class_name] += 1

        print(f"Class distribution: {dict(final_class_counts)}")

        return samples
    
    def _sample_subset(self, samples: List[Tuple[Path, int]], class_counts: Dict[str, int]) -> List[Tuple[Path, int]]:
        """
        Sample a subset of training data while maintaining perfect class balance.
        
        Args:
            samples: List of (image_path, class_index) tuples
            class_counts: Dictionary of class name to count
            
        Returns:
            Sampled subset of samples with equal number of samples per class
        """
        import random
        
        # Set random seed for reproducible sampling
        random.seed(self.random_seed)
        
        # Group samples by class
        samples_by_class = {cls: [] for cls in self.classes}
        for sample_path, class_idx in samples:
            class_name = self.classes[class_idx]
            samples_by_class[class_name].append((sample_path, class_idx))
        
        # Find minimum class count (should be same for all classes in OUHANDS)
        min_class_count = min(len(samples_by_class[cls]) for cls in self.classes)
        
        # Calculate target samples per class to maintain perfect balance
        target_samples_per_class = int(min_class_count * self.train_subset_ratio)
        
        # Ensure at least 1 sample per class if train_subset_ratio > 0
        if self.train_subset_ratio > 0 and target_samples_per_class == 0:
            target_samples_per_class = 1
            
        # Sample exactly the same number from each class
        sampled_samples = []
        for class_name in self.classes:
            class_samples = samples_by_class[class_name]
            
            # Randomly sample target_samples_per_class from this class
            if target_samples_per_class < len(class_samples):
                sampled_class_samples = random.sample(class_samples, target_samples_per_class)
            else:
                sampled_class_samples = class_samples
                
            sampled_samples.extend(sampled_class_samples)
        
        # Sort by filename for reproducible ordering
        sampled_samples.sort(key=lambda x: x[0].name)
        
        print(f"Balanced sampling: {target_samples_per_class} samples per class × {len(self.classes)} classes = {len(sampled_samples)} total samples")
        
        return sampled_samples
    
    def _load_bounding_box(self, filename: str) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Load bounding box for a given image filename.
        
        Args:
            filename: Image filename (e.g., 'A-ima-0001.png')
            
        Returns:
            Tuple of (x, y, width, height, confidence) or None if not found
        """
        if not self.use_bounding_box:
            return None
            
        # Map split names to actual directory paths
        if self.split in ['train', 'validation']:
            # Both train and validation use OUHANDS_train bounding boxes
            bbox_dir = self.root_dir / 'OUHANDS_train/train/hand_data/bounding_box'
        else:  # test split
            bbox_dir = self.root_dir / 'OUHANDS_test/test/hand_data/bounding_box'
        
        bbox_file = bbox_dir / f"{Path(filename).stem}.txt"
        
        if not bbox_file.exists():
            print(f"Warning: Bounding box file not found: {bbox_file}")
            return None
        
        try:
            with open(bbox_file, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) < 2:
                    return None
                
                # Parse bounding box data
                # Format: x y width height confidence
                bbox_data = lines[1].split()
                if len(bbox_data) >= 4:
                    x, y, width, height = map(int, bbox_data[:4])
                    confidence = float(bbox_data[4]) if len(bbox_data) > 4 else 1.0
                    return (x, y, width, height, confidence)
                
        except Exception as e:
            print(f"Error reading bounding box file {bbox_file}: {e}")
            
        return None
    
    def _crop_image_to_bbox(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop image to bounding box with optional padding.
        
        Args:
            image: PIL Image
            bbox: Tuple of (x, y, width, height)
            
        Returns:
            Cropped PIL Image
        """
        x, y, width, height = bbox
        
        # Add padding
        padding = self.bbox_padding
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.width, x + width + padding)
        y_max = min(image.height, y + height + padding)
        
        # Crop the image
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        return cropped_image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            If use_bounding_box=False: (image, label) or (image, label, path)
            If use_bounding_box=True: (image, label, bbox) or (image, label, bbox, path)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        image_path, label = self.samples[idx]
        filename = image_path.name
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Load bounding box if requested
        bbox = None
        if self.use_bounding_box:
            bbox_data = self._load_bounding_box(filename)
            if bbox_data is not None:
                bbox = bbox_data[:4]  # (x, y, width, height)
                
                # Crop image to bounding box if requested
                if self.crop_to_bbox:
                    image = self._crop_image_to_bbox(image, bbox)
                    # Adjust bbox coordinates after cropping (relative to cropped image)
                    x, y, width, height = bbox
                    padding = self.bbox_padding
                    x_min = max(0, x - padding)
                    y_min = max(0, y - padding)
                    # New bbox coordinates in cropped image
                    bbox = (x - x_min, y - y_min, width, height)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        # Return data based on configuration
        if self.use_bounding_box:
            if self.return_paths:
                return image, label, bbox, str(image_path)
            else:
                return image, label, bbox
        else:
            if self.return_paths:
                return image, label, str(image_path)
            else:
                return image, label
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index."""
        if class_idx < 0 or class_idx >= len(self.classes):
            raise ValueError(f"Class index {class_idx} out of range [0, {len(self.classes)-1}]")
        return self.classes[class_idx]

    def get_class_counts(self) -> Dict[str, int]:
        """Get the number of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            counts[class_name] += 1
        return counts

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        image_path, label = self.samples[idx]
        info = {
            'index': idx,
            'path': str(image_path),
            'filename': image_path.name,
            'class_idx': label,
            'class_name': self.classes[label],
            'split': self.split
        }
        
        # Add bounding box info if available
        if self.use_bounding_box:
            bbox_data = self._load_bounding_box(image_path.name)
            if bbox_data is not None:
                info['bbox'] = {
                    'x': bbox_data[0],
                    'y': bbox_data[1], 
                    'width': bbox_data[2],
                    'height': bbox_data[3],
                    'confidence': bbox_data[4]
                }
            else:
                info['bbox'] = None
                
        return info

    def filter_by_classes(self, classes: List[str]) -> 'OuhandsDS':
        """
        Create a new dataset instance with only specified classes.
        
        Args:
            classes: List of class names to keep
            
        Returns:
            New OuhandsDS instance with filtered classes
        """
        return OuhandsDS(
            root_dir=str(self.root_dir),
            split=self.split,
            transform=self.transform,
            target_transform=self.target_transform,
            return_paths=self.return_paths,
            class_subset=classes,
            use_bounding_box=self.use_bounding_box,
            crop_to_bbox=self.crop_to_bbox,
            bbox_padding=self.bbox_padding,
            train_subset_ratio=self.train_subset_ratio,
            random_seed=self.random_seed
        )
    
    @staticmethod
    def get_default_transforms(image_size: int = 224, normalize: bool = True) -> transforms.Compose:
        """
        Get commonly used transforms for OUHANDS dataset.

        Args:
            image_size: Target image size (square)
            normalize: Whether to apply ImageNet normalization

        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            # ImageNet normalization (commonly used for pretrained models)
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        return transforms.Compose(transform_list)

    @staticmethod
    def get_augmentation_transforms(image_size: int = 224, normalize: bool = True) -> transforms.Compose:
        """
        Get augmentation transforms for training.

        Args:
            image_size: Target image size (square)
            normalize: Whether to apply ImageNet normalization

        Returns:
            Composed transforms with augmentation
        """
        transform_list = [
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ]

        if normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        return transforms.Compose(transform_list)

    @classmethod
    def create_label_efficiency_datasets(
        cls, 
        train_subset_ratio: float = 0.1,
        random_seed: int = 42,
        **kwargs
    ) -> Tuple['OuhandsDS', 'OuhandsDS', 'OuhandsDS']:
        """
        Convenience method to create train/val/test datasets for label efficiency experiments.
        
        Args:
            train_subset_ratio: Fraction of training data to use (0.0 to 1.0)
            random_seed: Random seed for reproducible sampling
            **kwargs: Additional arguments passed to OuhandsDS constructor
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
            
        Example:
            # Use only 10% of training data for label efficiency study
            train_ds, val_ds, test_ds = OuhandsDS.create_label_efficiency_datasets(
                train_subset_ratio=0.1,
                crop_to_bbox=True,
                use_bounding_box=True
            )
        """
        train_ds = cls(
            split='train', 
            train_subset_ratio=train_subset_ratio,
            random_seed=random_seed,
            **kwargs
        )
        
        val_ds = cls(
            split='validation',
            random_seed=random_seed,
            **kwargs
        )
        
        test_ds = cls(
            split='test',
            random_seed=random_seed,
            **kwargs
        )
        
        return train_ds, val_ds, test_ds
