import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights 

class MobileNetV2Landmark(nn.Module):
    def __init__(self, pretrained: bool = False, num_landmarks: int = 21):
        """
        MobileNetV2-based landmark regression model.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_landmarks: Number of landmarks (21 for hand)
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Use torchvision's MobileNetV2 backbone
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # Replace classifier with landmark regression head
        in_features = self.backbone.classifier[-1].in_features  # 1280
        
        # More robust regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), 
            nn.Linear(512, num_landmarks * 3),  # 3D landmarks (x, y, z)
        )
        
        # Initialize the new layers
        self._initialize_head()
    
    def _initialize_head(self):
        """Initialize the regression head with proper weights."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            landmarks: Predicted landmarks [B, num_landmarks * 3]
        """
        return self.backbone(x)