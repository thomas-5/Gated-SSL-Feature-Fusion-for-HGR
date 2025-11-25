"""
SwavDualModel: A Dual-Stream SwAV model (Global + Local) inspired by the logic in
DINO_BBox_Focus_Experiment.ipynb.

Logic:
1. Global View: Full image resized to 224x224 -> SwAV -> Feature
2. Local View: Image cropped to BBox and resized to 224x224 -> SwAV -> Feature
3. Fusion: Concatenate features -> Classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_model import crop_and_resize

class SwavDualModel(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        # Load SwAV ResNet50
        self.backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=pretrained)
        
        # Remove original fc
        self.backbone.fc = nn.Identity()
        
        # ResNet50 feature dim is 2048
        self.feature_dim = 2048
        
        # Fusion: Global (2048) + Local (2048) = 4096
        self.fusion_dim = self.feature_dim * 2
        
        # Classifier structure from the notebook
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor, gt_bboxes: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
            gt_bboxes: Optional Ground Truth Bounding Boxes (B, 4) in (x, y, w, h) format.
        """
        # 1. Global View
        # Resize to 224x224 if not already (SwAV expects standard size)
        # Note: If input is already 224x224, this is a no-op or just interpolation
        if x.shape[-2:] != (224, 224):
            x_global = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            x_global = x
            
        global_feat = self.backbone(x_global) # (B, 2048)
        
        # 2. Local View
        if gt_bboxes is not None:
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            if gt_bboxes.device != x.device:
                gt_bboxes = gt_bboxes.to(x.device)
            
            x_min = gt_bboxes[:, 0]
            y_min = gt_bboxes[:, 1]
            w_box = gt_bboxes[:, 2]
            h_box = gt_bboxes[:, 3]
            
            x_max = x_min + w_box
            y_max = y_min + h_box
            
            bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
            
            # Crop and resize to 224x224
            x_local = crop_and_resize(x, bboxes, target_size=(224, 224))
        else:
            # Fallback: Use global view as local view if no bbox
            x_local = x_global
            
        local_feat = self.backbone(x_local) # (B, 2048)
        
        # 3. Fusion
        fused = torch.cat([global_feat, local_feat], dim=1) # (B, 4096)
        
        # 4. Classification
        logits = self.classifier(fused)
        
        return logits
