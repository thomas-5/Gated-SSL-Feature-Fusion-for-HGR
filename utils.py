"""Shared utilities for training and evaluation workflows."""
from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import apply_rot_embed_cat


def select_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_add_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Match timm's mask application semantics for attention tensors."""
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
    """Custom attention forward that also returns attention weights."""
    B, N, C = x.shape

    if getattr(attn_module, "qkv", None) is not None:
        if attn_module.q_bias is None:
            qkv = attn_module.qkv(x)
        else:
            qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
            if attn_module.qkv_bias_separate:
                qkv = attn_module.qkv(x)
                qkv = qkv + qkv_bias
            else:
                qkv = F.linear(x, weight=attn_module.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
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
    attn_dropped = attn_module.attn_drop(attn)

    x = attn_dropped @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = attn_module.norm(x)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x, attn


def _forward_block_with_attention(
    block: nn.Module,
    x: torch.Tensor,
    rope: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward a transformer block while capturing attention weights."""
    attn_out, attn_weights = _attention_forward_with_weights(block.attn, block.norm1(x), rope=rope)

    if block.gamma_1 is None:
        x = x + block.drop_path1(attn_out)
        x = x + block.drop_path2(block.mlp(block.norm2(x)))
    else:
        x = x + block.drop_path1(block.gamma_1 * attn_out)
        x = x + block.drop_path2(block.gamma_2 * block.mlp(block.norm2(x)))

    return x, attn_weights


def forward_with_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    return_cls: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass returning logits, attention weights, and optionally CLS tokens."""
    x = model.patch_embed(images)
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

        if idx == len(model.blocks) - 1:
            x, attn_weights = _forward_block_with_attention(block, x, rope)
        else:
            x = block(x, rope=rope)

    x = model.norm(x)
    logits = model.forward_head(x)
    if return_cls:
        cls_token = x[:, 0].contiguous()
        return logits, attn_weights, cls_token
    return logits, attn_weights


def model_forward_with_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    return_cls: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch attention-aware forward pass supporting wrapped models."""
    if hasattr(model, "forward_with_attention"):
        return model.forward_with_attention(images, return_cls=return_cls)
    return forward_with_attention(model, images, return_cls=return_cls)
