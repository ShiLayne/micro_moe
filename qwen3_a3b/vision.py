"""Vision tower + alignment module for the multimodal Qwen3 variant.

Two pieces:

:class:`Qwen3VisionModel`
    A CLIP-style ViT: Conv2d patch embedding, learned 1D position embedding, pre-LayerNorm
    transformer blocks with bidirectional attention, final LayerNorm. Defaults describe
    **ViT-S/16** (384 wide, 12 blocks, 6 heads, 224x224 input) at 21.66M parameters.

:class:`VisionAligner`
    Linear projection from the vision width into the language width, followed by
    ``align_num_hidden_layers`` transformer blocks (default: exactly one) operating at the
    language width, closing with an **RMSNorm**. This is what makes visual features speak
    the decoder's language.

Why the two halves normalise differently
----------------------------------------
The ViT follows the CLIP convention (LayerNorm with weight *and* bias, learned absolute
positions) because that is what the released vision checkpoints use -- keeping it
bit-compatible lets the tower be warm-started from a pretrained CLIP/SigLIP encoder.
The aligner, by contrast, lives at the decoder's width and feeds its residual stream, so it
uses the decoder's own RMSNorm (weight only, no bias, no mean subtraction). That keeps the
two halves consistent about statistics without breaking vision-checkpoint compatibility.

How vision reaches the language model
-------------------------------------
Image features are **not** appended to the text sequence. Following the Qwen-VL family,
the decoder reserves ``num_patches`` positions holding ``image_token_id``, and
:meth:`Qwen3VisionModel.extract_image_features` splices the projected patch features into
the text embedding tensor at those positions. That keeps the attention mask, position ids
and KV cache layout identical to the text-only model.

Both the ViT and the aligner use a standard 2-matrix GELU MLP (not the decoder's SwiGLU).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3MoeConfig, VisionConfig
from .layers import RMSNorm

__all__ = [
    "NormKind",
    "Qwen3VisionModel",
    "VisionAligner",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
]

#: Which normalisation a :class:`VisionBlock` uses.
NormKind = str  # "layer_norm" | "rms_norm"


def _make_norm(kind: NormKind, dim: int, eps: float) -> nn.Module:
    """Build the block's normalisation layer.

    ``layer_norm`` has weight + bias and subtracts the mean; ``rms_norm`` has weight only and
    leaves the mean alone. The distinction is real for parameter counting and
    initialisation, so it is threaded through as an explicit option rather than inferred.
    """
    if kind == "rms_norm":
        return RMSNorm(dim, eps=eps)
    if kind == "layer_norm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind {kind!r}; expected 'layer_norm' or 'rms_norm'")


class VisionMLP(nn.Module):
    """Standard 2-matrix GELU MLP, as used by CLIP ViT (not SwiGLU)."""

    def __init__(self, dim: int, intermediate_size: int, hidden_act: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, dim)
        self.act = F.gelu if hidden_act.startswith("gelu") else F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class VisionAttention(nn.Module):
    """Full (bidirectional) multi-head self-attention with biases.

    Images have no causal structure, so ``is_causal`` is never set: every patch attends to
    every other patch. One fused ``qkv`` projection keeps the parameter layout close to the
    CLIP checkpoints that vision towers are usually initialised from.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args: ``(batch, seq, dim)``; mask is additive/bool, never causal."""
        batch, seq_len, dim = hidden_states.shape
        qkv = self.qkv(hidden_states)
        # (b, s, 3, heads, head_dim) -> (3, b, heads, s, head_dim)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out_proj(attn_output)


class VisionBlock(nn.Module):
    """Pre-norm transformer block shared by the ViT tower and the aligner.

    Used at two different widths *and* two normalisations: inside the ViT it runs at
    ``vision.hidden_size`` with LayerNorm (CLIP convention), inside the aligner at the
    *language* ``hidden_size`` with RMSNorm (decoder convention). Parameter counts are
    ``block(dim, intermediate_size, "layer_norm")`` and ``block(hidden_size, 4 * hidden_size,
    "rms_norm")`` respectively -- both reflected in
    ``Qwen3MoeConfig.estimate_vision_parameters``.
    """

    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
        hidden_act: str = "gelu",
        attention_dropout: float = 0.0,
        norm_kind: NormKind = "layer_norm",
    ) -> None:
        super().__init__()
        self.norm_kind = norm_kind
        self.norm1 = _make_norm(norm_kind, dim, eps)
        self.attn = VisionAttention(dim, num_heads, attention_dropout)
        self.norm2 = _make_norm(norm_kind, dim, eps)
        self.mlp = VisionMLP(dim, intermediate_size, hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VisionModel(nn.Module):
    """CLIP-style ViT tower producing one feature per patch token.

    Normalisation is LayerNorm and positions are a learned absolute table, matching the
    released CLIP/SigLIP encoders this tower is designed to be warm-started from.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size

        # Patch embedding as a strided Conv2d: (b, 3, 224, 224) -> (b, dim, 14, 14).
        self.patch_embed = nn.Conv2d(
            config.num_channels, dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        # Learned absolute positions: one vector per patch, no class token.
        self.position_ids = nn.Parameter(
            torch.zeros(1, config.num_patches, dim), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            VisionBlock(
                dim=dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                eps=config.layer_norm_eps,
                hidden_act=config.hidden_act,
                attention_dropout=config.attention_dropout,
                norm_kind="layer_norm",
            )
            for _ in range(config.num_hidden_layers)
        )
        self.post_layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, 3, H, W)`` float pixels in ``[0, 1]`` or normalised.

        Returns:
            ``(batch, num_patches, vision.hidden_size)`` -- 196 x 384 for ViT-S/16.
        """
        expected = self.config.image_size
        if pixel_values.dim() != 4 or pixel_values.shape[1] != self.config.num_channels:
            raise ValueError(
                "pixel_values must be (batch, channels, height, width) with "
                f"{self.config.num_channels} channels, got {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[-2:] != (expected, expected):
            raise ValueError(
                f"pixel_values must be {expected}x{expected}, got "
                f"{tuple(pixel_values.shape[-2:])}. Resize in the data pipeline; the tower "
                "has a fixed learned position embedding."
            )

        # (b, dim, 14, 14) -> (b, 196, dim); row-major, matching the position table.
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_ids.to(hidden_states.dtype)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        return self.post_layernorm(hidden_states)


class VisionAligner(nn.Module):
    """Project vision features into the language width, then refine them.

    Composition (defaults)::

        proj      Linear(384 -> 1024)            vision width -> language width
        blocks    1 x transformer block(1024)     bidirectional, 4x MLP, **RMSNorm**
        norm      RMSNorm(1024)                   weight only, no bias

    The single transformer block is deliberate: it gives the alignment enough capacity to
    learn a cross-modal mapping without competing with the decoder for budget. Raise
    ``config.vision.align_num_hidden_layers`` to stack more.

    No position encoding is added here. The tower already emits position-aware features, and
    the aligner's job is a per-token channel mapping plus local mixing, so absolute
    positions would only re-encode information the tower already resolved.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("VisionAligner requires a vision config")
        lang_dim = config.hidden_size
        self.vision_dim = vision.hidden_size
        self.lang_dim = lang_dim

        self.proj = nn.Linear(vision.hidden_size, lang_dim)
        self.blocks = nn.ModuleList(
            VisionBlock(
                dim=lang_dim,
                intermediate_size=4 * lang_dim,
                num_heads=config.num_attention_heads,
                eps=config.rms_norm_eps,
                hidden_act=vision.hidden_act,
                attention_dropout=vision.attention_dropout,
                norm_kind="rms_norm",
            )
            for _ in range(vision.align_num_hidden_layers)
        )
        self.norm = RMSNorm(lang_dim, eps=config.rms_norm_eps)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Args/returns: ``(batch, num_patches, vision_dim)`` -> ``(..., lang_dim)``."""
        hidden_states = self.proj(vision_features)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.norm(hidden_states)
