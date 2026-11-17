"""A from-scratch PyTorch implementation of Qwen3-30B-A3B (``qwen3_moe``).

Public surface
--------------
Config
    :class:`Qwen3MoeConfig`, :func:`qwen3_30b_a3b_config`, :func:`qwen3_2b_a600m_config`,
    :func:`qwen3_tiny_debug_config`, :func:`build_config`, :class:`ParamBreakdown`

Model
    :class:`Qwen3MoeForCausalLM` (trunk + LM head + losses), :class:`Qwen3MoeModel`

Layers / MoE
    :class:`RMSNorm`, :class:`RotaryEmbedding`, :class:`Qwen3MoeAttention`,
    :class:`Qwen3MoeMLP`, :class:`TopKRouter`, :class:`GroupedExperts`,
    :class:`SparseMoeBlock`

Losses
    :func:`chunked_linear_cross_entropy`, :func:`global_load_balancing_loss`,
    :func:`router_z_loss`, :func:`compute_moe_losses`

Training helpers
    :func:`build_param_groups`, :func:`apply_fsdp2`

The pretraining entry point is ``train.py`` at the repo root::

    python train.py --config qwen3-tiny-debug --dry-run
    torchrun --standalone --nproc_per_node=8 train.py --config qwen3-30b-a3b \
        --data /path/to/tokens.bin --fsdp2 --gradient-checkpointing
"""

from __future__ import annotations

from .config import (
    CONFIG_PRESETS,
    ParamBreakdown,
    Qwen3MoeConfig,
    VisionConfig,
    build_config,
    human_count,
    qwen3_2b_a600m_config,
    qwen3_30b_a3b_config,
    qwen3_a300m_config,
    qwen3_a300m_vision_config,
    qwen3_tiny_debug_config,
)
from .layers import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_causal_mask,
    build_document_mask,
    mask_value,
    repeat_kv,
    rotate_half,
    supports_gqa_sdpa,
)
from .loss import (
    IGNORE_INDEX,
    MoeLossOutput,
    all_reduce_sum,
    assignment_counts,
    chunked_linear_cross_entropy,
    compute_moe_losses,
    expert_load_metrics,
    global_load_balancing_loss,
    router_z_loss,
)
from .model import (
    CausalLMOutput,
    KVCache,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    apply_fsdp2,
    build_param_groups,
)
from .moe import (
    GroupedExperts,
    SparseMoeBlock,
    TopKRouter,
    grouped_mm_available,
)
from .vision import (
    NormKind,
    Qwen3VisionModel,
    VisionAligner,
    VisionAttention,
    VisionBlock,
    VisionMLP,
)

__version__ = "0.1.0"

__all__ = [
    "CONFIG_PRESETS",
    "IGNORE_INDEX",
    "CausalLMOutput",
    "GroupedExperts",
    "KVCache",
    "MoeLossOutput",
    "NormKind",
    "ParamBreakdown",
    "Qwen3MoeAttention",
    "Qwen3MoeConfig",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeMLP",
    "Qwen3MoeModel",
    "Qwen3VisionModel",
    "RMSNorm",
    "RotaryEmbedding",
    "SparseMoeBlock",
    "TopKRouter",
    "VisionAligner",
    "VisionAttention",
    "VisionBlock",
    "VisionConfig",
    "VisionMLP",
    "__version__",
    "all_reduce_sum",
    "apply_fsdp2",
    "apply_rotary_pos_emb",
    "assignment_counts",
    "build_causal_mask",
    "build_config",
    "build_document_mask",
    "build_param_groups",
    "chunked_linear_cross_entropy",
    "compute_moe_losses",
    "expert_load_metrics",
    "global_load_balancing_loss",
    "grouped_mm_available",
    "human_count",
    "mask_value",
    "qwen3_2b_a600m_config",
    "qwen3_30b_a3b_config",
    "qwen3_a300m_config",
    "qwen3_a300m_vision_config",
    "qwen3_tiny_debug_config",
    "repeat_kv",
    "rotate_half",
    "router_z_loss",
    "supports_gqa_sdpa",
]
