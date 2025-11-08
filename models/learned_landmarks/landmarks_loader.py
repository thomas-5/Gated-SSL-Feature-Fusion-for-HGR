import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union

class OuhandsLandmarks(Dataset):
    def __init__(
        self,
        split: str = 'train',  # 'train', 'validation', 'test'
        dataset_root: str = 'dataset',
        landmarks_root: str = 'landmarks', 
        image_size: int = 224,
        augment: bool = True,
        standardize: bool = True,
    ):
        """
        Dataset for OUHANDS hand landmark regression using MediaPipe landmarks as ground truth.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            dataset_root: Root directory of OUHANDS dataset  
            landmarks_root: Root directory of extracted landmarks
            image_size: Target image size for resizing
            augment: Whether to apply data augmentation (only for training)
            standardize: Whether to standardize landmark coordinates to [-1, 1]
        """
        super().__init__()
        self.split = split
        self.dataset_root = dataset_root
        self.landmarks_root = landmarks_root
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.standardize = standardize
        
        # Load data items (image_path, landmarks)
        self.items = self._load_data_items()
        
        print(f"Loaded {len(self.items)} samples for {split} split")
        
        # Define transforms
        self._setup_transforms()
    
    def _load_data_items(self) -> List[Tuple[str, np.ndarray]]:
        """Load valid data items with both image and landmarks."""
        items = []
        
        # Map split names
        split_map = {
            'train': 'train',
            'validation': 'validation', 
            'test': 'test'
        }
        
        # Handle relative paths - look from project root
        if not os.path.isabs(self.landmarks_root):
            # Go up two levels from models/learned_landmarks to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            landmarks_root = os.path.join(project_root, self.landmarks_root)
        else:
            landmarks_root = self.landmarks_root
            
        landmarks_split_dir = os.path.join(landmarks_root, split_map[self.split])
        
        if not os.path.exists(landmarks_split_dir):
            raise FileNotFoundError(f"Landmarks directory not found: {landmarks_split_dir}")
            
        print(f"Loading from: {landmarks_split_dir}")
        
        # Get all landmark files for this split
        landmark_files = [f for f in os.listdir(landmarks_split_dir) if f.endswith('_landmarks.json')]
        
        for landmark_file in landmark_files:
            landmark_path = os.path.join(landmarks_split_dir, landmark_file)
            
            try:
                # Load landmark data
                with open(landmark_path, 'r') as f:
                    landmark_data = json.load(f)
                
                # Skip if no landmarks detected
                if not landmark_data.get('landmarks'):
                    continue
                
                # Get image path
                image_path = landmark_data['image_path']
                
                # Convert to absolute path if relative
                if not os.path.isabs(image_path):
                    # Assuming the path in JSON is relative to project root
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    image_path = os.path.join(project_root, image_path)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    continue
                
                # Extract landmark coordinates
                landmarks = landmark_data['landmarks']
                if len(landmarks) != 21:  # MediaPipe hand has 21 landmarks
                    continue
                
                # Flatten landmarks to (x1, y1, z1, x2, y2, z2, ...) format
                coords = []
                for lm in landmarks:
                    coords.extend([lm['x'], lm['y'], lm['z']])
                
                coords = np.array(coords, dtype=np.float32)  # Shape: [63]
                
                items.append((image_path, coords))
                
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Warning: Failed to load {landmark_file}: {e}")
                continue
        
        return items
    
    def _setup_transforms(self):
        """Setup image transformations."""
        # Base transforms (always applied)
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        # Add augmentation for training
        if self.augment:
            # Note: We use geometry-preserving augmentations to avoid 
            # needing to transform landmark coordinates
            augment_transforms = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
            ]
            base_transforms = augment_transforms + base_transforms
        
        # Normalization
        base_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225]))
        
        self.transform = transforms.Compose(base_transforms)
    
    def _standardize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Standardize landmark coordinates to [-1, 1] range.
        
        Args:
            landmarks: Raw landmark coordinates [63] (x1,y1,z1, x2,y2,z2, ...)
            
        Returns:
            Standardized landmarks [63]
        """
        if not self.standardize:
            return landmarks
        
        # Reshape to (21, 3) for easier processing
        coords = landmarks.reshape(21, 3)
        
        # Standardize x and y coordinates (already in [0,1] range from MediaPipe)
        # Map from [0,1] to [-1,1]
        coords[:, 0] = coords[:, 0] * 2.0 - 1.0  # x coordinates
        coords[:, 1] = coords[:, 1] * 2.0 - 1.0  # y coordinates
        
        # For z coordinates, they're relative depth values, normalize by std
        z_coords = coords[:, 2]
        if len(z_coords) > 1:
            z_std = np.std(z_coords)
            if z_std > 0:
                coords[:, 2] = z_coords / (z_std + 1e-8)
        
        return coords.flatten()

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image_path, landmarks = self.items[idx]
        
        # Load and transform image
        try:
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image and landmarks
            img = torch.zeros(3, self.image_size, self.image_size)
        
        # Process landmarks
        landmarks = self._standardize_landmarks(landmarks)
        target = torch.from_numpy(landmarks)  # [63]
        
        return img, target