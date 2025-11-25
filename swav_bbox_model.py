"""
SwavBBoxModel: A SwAV-based model that uses bounding boxes to crop the input image
before feeding it to the network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_model import crop_and_resize

class SwavBBoxModel(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        # Load SwAV ResNet50
        # We use the same hub loading as in fusion_model.py
        self.swav = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=pretrained)
        
        # The SwAV model from torch.hub has a 'fc' layer.
        # We replace it with our specific classifier.
        # ResNet50 feature dim is 2048.
        self.swav.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x: torch.Tensor, gt_bboxes: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
            gt_bboxes: Optional Ground Truth Bounding Boxes (B, 4) in (x, y, w, h) format.
        """
        if gt_bboxes is not None:
            # Convert (x, y, w, h) to (x1, y1, x2, y2) for crop_and_resize
            if gt_bboxes.device != x.device:
                gt_bboxes = gt_bboxes.to(x.device)
            
            x_min = gt_bboxes[:, 0]
            y_min = gt_bboxes[:, 1]
            w_box = gt_bboxes[:, 2]
            h_box = gt_bboxes[:, 3]
            
            x_max = x_min + w_box
            y_max = y_min + h_box
            
            bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
            
            # Crop and resize to 224x224 (standard ResNet input)
            # crop_and_resize handles the batch loop and interpolation
            x = crop_and_resize(x, bboxes, target_size=(224, 224))
        else:
            # If no bbox is provided (e.g. during simple inference without detection),
            # we resize the full image to 224x224.
            # Note: In a real pipeline, you'd want a detector here.
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        return self.swav(x)
