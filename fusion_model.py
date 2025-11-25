"""
Fusion model combining Segmentation-Enhanced DINO and SwAV via Gated Mechanism.
Implements the pipeline:
1. DINO -> Attention Map -> Bounding Box
2. Crop Image -> SwAV -> Hand Features
3. DINO Features + SwAV Features -> Gated Fusion -> Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from typing import Tuple, Optional, List

# Import utils to reuse rope/attention logic if needed, 
# but we will likely reimplement the forward loop to get features instead of logits.
from timm.layers import apply_rot_embed_cat

def _maybe_add_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attn_mask is None:
        return attn
    if attn_mask.dtype == torch.bool:
        return attn.masked_fill(~attn_mask, float("-inf"))
    return attn + attn_mask

def _attention_forward_with_weights(
    attn_module: nn.Module,
    x: torch.Tensor,
    rope: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modified from utils.py to be self-contained or we could import.
    Extracts attention weights from a timm Attention module.
    """
    B, N, C = x.shape

    # Handle different timm versions/architectures of Attention
    if getattr(attn_module, "qkv", None) is not None:
        # Standard ViT Attention
        if attn_module.q_bias is None:
            qkv = attn_module.qkv(x)
        else:
            qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
            if getattr(attn_module, "qkv_bias_separate", False):
                qkv = attn_module.qkv(x)
                qkv = qkv + qkv_bias
            else:
                qkv = F.linear(x, weight=attn_module.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        # Some newer timm models use separate q/k/v projs
        q = attn_module.q_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)
        k = attn_module.k_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)
        v = attn_module.v_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)

    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if rope is not None:
        num_prefix = attn_module.num_prefix_tokens
        half = getattr(attn_module, "rotate_half", False)
        q_suffix = apply_rot_embed_cat(q[:, :, num_prefix:, :], rope, half=half)
        k_suffix = apply_rot_embed_cat(k[:, :, num_prefix:, :], rope, half=half)
        q = torch.cat([q[:, :, :num_prefix, :], q_suffix.type_as(v)], dim=2)
        k = torch.cat([k[:, :, :num_prefix, :], k_suffix.type_as(v)], dim=2)

    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)
    attn = _maybe_add_mask(attn, attn_mask)
    attn = attn.softmax(dim=-1)
    
    # Save weights before dropout
    weights = attn
    
    attn = attn_module.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = attn_module.norm(x)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x, weights

def get_dino_features_and_attention(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through DINO ViT to get:
    1. CLS token features (before head)
    2. Attention weights from the last block
    """
    x = model.patch_embed(x)
    x, rot_pos_embed = model._pos_embed(x)
    x = model.norm_pre(x)

    attn_weights = None

    for idx, block in enumerate(model.blocks):
        if rot_pos_embed is None:
            rope = None
        elif getattr(model, "rope_mixed", False) and rot_pos_embed is not None:
            rope = rot_pos_embed[idx]
        else:
            rope = rot_pos_embed

        # If last block, capture attention
        if idx == len(model.blocks) - 1:
            # Manual forward of the block to capture attention
            # block(x) usually does: x + drop_path(attn(norm1(x))) + drop_path(mlp(norm2(x)))
            
            # 1. Attention part
            norm_x = block.norm1(x)
            attn_out, weights = _attention_forward_with_weights(block.attn, norm_x, rope=rope)
            attn_weights = weights
            
            # Apply DropPath and Residual
            if block.gamma_1 is None:
                x = x + block.drop_path1(attn_out)
            else:
                x = x + block.drop_path1(block.gamma_1 * attn_out)
            
            # 2. MLP part
            norm_x2 = block.norm2(x)
            mlp_out = block.mlp(norm_x2)
            
            if block.gamma_2 is None:
                x = x + block.drop_path2(mlp_out)
            else:
                x = x + block.drop_path2(block.gamma_2 * mlp_out)
                
        else:
            x = block(x, rope=rope)

    x = model.norm(x)
    # x is (B, N, C), we want CLS token (index 0)
    cls_features = x[:, 0]
    
    return cls_features, attn_weights

def get_attention_bboxes(
    attn_weights: torch.Tensor, 
    img_size: Tuple[int, int], 
    patch_size: int = 16,
    threshold: float = 0.6,
    margin: float = 0.1
) -> torch.Tensor:
    """
    Convert attention weights to bounding boxes.
    
    Args:
        attn_weights: (B, Heads, N, N)
        img_size: (H, W)
        patch_size: ViT patch size
        threshold: Threshold relative to max attention to define foreground
        margin: Margin to add around bbox (fraction of size)
        
    Returns:
        bboxes: (B, 4) in (x1, y1, x2, y2) format
    """
    B, Heads, N, _ = attn_weights.shape
    H, W = img_size
    
    # 3. Reshape to grid
    grid_h, grid_w = H // patch_size, W // patch_size
    num_patches = grid_h * grid_w
    
    # 1. Extract CLS attention: (B, Heads, num_patches)
    # We take the last num_patches tokens to handle potential register tokens (prefix)
    cls_attn = attn_weights[:, :, 0, -num_patches:] 
    
    # 2. Average over heads: (B, num_patches)
    cls_attn = cls_attn.mean(dim=1)
    
    cls_attn = cls_attn.reshape(B, 1, grid_h, grid_w)
    
    # 4. Upsample to image size
    cls_attn = F.interpolate(cls_attn, size=(H, W), mode='bilinear', align_corners=False)
    cls_attn = cls_attn.squeeze(1) # (B, H, W)
    
    bboxes = []
    
    for i in range(B):
        attn_map = cls_attn[i]
        
        # Thresholding
        # Use a dynamic threshold based on the max value in the map
        max_val = attn_map.max()
        if max_val < 1e-6:
            # Fallback to full image if no attention
            bboxes.append(torch.tensor([0, 0, W, H], device=attn_weights.device))
            continue
            
        mask = attn_map > (threshold * max_val)
        
        # Find coords
        nonzero = mask.nonzero()
        if nonzero.size(0) == 0:
             bboxes.append(torch.tensor([0, 0, W, H], device=attn_weights.device))
             continue
             
        # nonzero is (num_points, 2) -> (y, x)
        y_min = nonzero[:, 0].min().item()
        y_max = nonzero[:, 0].max().item()
        x_min = nonzero[:, 1].min().item()
        x_max = nonzero[:, 1].max().item()
        
        box_h = y_max - y_min
        box_w = x_max - x_min
        
        # Add margin
        pad_h = int(box_h * margin)
        pad_w = int(box_w * margin)
        
        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(W, x_max + pad_w)
        y_max = min(H, y_max + pad_h)
        
        bboxes.append(torch.tensor([x_min, y_min, x_max, y_max], device=attn_weights.device))
        
    return torch.stack(bboxes)

def crop_and_resize(
    images: torch.Tensor, 
    bboxes: torch.Tensor, 
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Crop images based on bboxes and resize to target_size.
    """
    crops = []
    for i in range(images.size(0)):
        img = images[i]
        box = bboxes[i]
        x1, y1, x2, y2 = box.long()
        
        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            # Fallback: use full image if crop is invalid
            # print(f"Warning: Invalid crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}. Using full image.")
            raise RuntimeError(f"Input and output sizes should be greater than 0. Invalid crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            crop = img 
        else:
            crop = img[:, y1:y2, x1:x2]
            
        # Double check if crop is empty (e.g. if coordinates were out of bounds)
        if crop.numel() == 0 or crop.shape[1] == 0 or crop.shape[2] == 0:
             # print(f"Warning: Empty crop. Original coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}. Using full image.")
             raise RuntimeError(f"Input and output sizes should be greater than 0. Empty crop. Original coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
             crop = img

        crop = F.interpolate(crop.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        crops.append(crop.squeeze(0))
        
    return torch.stack(crops)


class DinoSwavFusionModel(nn.Module):
    def __init__(
        self, 
        dino_model_name: str = 'vit_small_patch16_224_dino',
        num_classes: int = 10,
        common_dim: int = 512,
        pretrained_dino: bool = True,
        pretrained_swav: bool = True
    ):
        super().__init__()
        
        # 1. DINO Backbone (Localization Assistant)
        # We use timm to load DINO. 
        # Note: 'vit_small_patch16_224_dino' is available in timm.
        # Add drop_path_rate for regularization
        self.dino = timm.create_model(dino_model_name, pretrained=pretrained_dino, drop_path_rate=0.1)
        self.dino_dim = self.dino.embed_dim
        
        # 2. SwAV Backbone (Feature Extractor for Global & Local)
        # Loading ResNet50 SwAV from torch.hub
        # We assume the user has internet access or cached weights.
        # If not, they should provide a path, but for now we use hub.
        if pretrained_swav:
            self.swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        else:
            self.swav = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=False)
            
        # Remove SwAV classification head (fc)
        self.swav.fc = nn.Identity()
        self.swav_dim = 2048 # ResNet50 output
        
        # 3. Projections
        # Global Stream (SwAV + DINO)
        # We concatenate SwAV and DINO features before projection
        self.proj_global = nn.Linear(self.swav_dim + self.dino_dim, common_dim)
        # Local Stream (SwAV)
        self.proj_local = nn.Linear(self.swav_dim, common_dim)
        
        # 4. Gating Mechanism
        # Input: [u_global; u_local] -> Gate g
        self.gate_net = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, common_dim),
            nn.Sigmoid()
        )
        
        # 5. Classifier
        self.classifier = nn.Linear(common_dim, num_classes)
        
    def forward(self, x: torch.Tensor, gt_bboxes: Optional[torch.Tensor] = None, return_attn: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
            gt_bboxes: Optional Ground Truth Bounding Boxes (B, 4) in (x, y, w, h) format.
                       If provided, these are used for cropping SwAV inputs (Training).
                       If None, DINO attention is used to generate boxes (Inference).
            return_attn: Whether to return attention weights (for auxiliary loss)
        Returns:
            logits: (B, num_classes)
            (optional) attn_weights: (B, Heads, N, N)
        """
        B, C, H, W = x.shape
        
        # --- Step 1: DINO Forward (Localization + Global Features) ---
        # Get attention map for bbox generation AND global features
        h_d_raw, attn_weights = get_dino_features_and_attention(self.dino, x)
        
        # --- Step 2: Cropping (GT or Attention) ---
        if gt_bboxes is not None:
            # print("Using Ground Truth Bounding Boxes")
            # Use Ground Truth if provided (Training)
            # OuhandsDS returns (x, y, w, h), we need (x1, y1, x2, y2)
            if gt_bboxes.device != x.device:
                gt_bboxes = gt_bboxes.to(x.device)
                
            x_min = gt_bboxes[:, 0]
            y_min = gt_bboxes[:, 1]
            w_box = gt_bboxes[:, 2]
            h_box = gt_bboxes[:, 3]
            
            x_max = x_min + w_box
            y_max = y_min + h_box
            
            bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        else:
            # Use Attention Map if GT is missing (Inference)
            # Note: This step is non-differentiable w.r.t coordinates
            bboxes = get_attention_bboxes(attn_weights, (H, W))
        
        # --- Step 3: Global Stream (SwAV + DINO) ---
        # Input: Full Image
        h_swav_global = self.swav(x) # (B, 2048)
        
        # Concatenate SwAV and DINO features
        h_global_combined = torch.cat([h_swav_global, h_d_raw], dim=1) # (B, 2048 + 384)
        
        u_global = self.proj_global(h_global_combined) # (B, common_dim)

        # --- Step 4: SwAV Local Stream ---
        # Input: Cropped Image
        # SwAV expects standard ImageNet size (224x224)
        x_crops = crop_and_resize(x, bboxes, target_size=(224, 224))
        h_local_raw = self.swav(x_crops) # (B, 2048)
        u_local = self.proj_local(h_local_raw) # (B, common_dim)
        
        # --- Step 5: Fusion ---
        # Concatenate
        combined = torch.cat([u_global, u_local], dim=1) # (B, 2*common_dim)
        
        # Compute Gate
        g = self.gate_net(combined) # (B, common_dim)
        
        # Fuse: u_fused = g * u_global + (1-g) * u_local
        u_fused = g * u_global + (1 - g) * u_local
        
        # --- Step 6: Classification ---
        logits = self.classifier(u_fused)
        
        if return_attn:
            return logits, attn_weights
            
        return logits

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to visualize attention or use for loss."""
        _, attn_weights = get_dino_features_and_attention(self.dino, x)
        return attn_weights

