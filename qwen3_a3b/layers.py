"""Primitive layers for a Qwen3-30B-A3B style decoder.

Contains only the pieces shared by every layer type:

* :class:`RMSNorm`            -- fp32 variance, no bias, no mean subtraction
* :class:`RotaryEmbedding`    -- standard RoPE plus optional YaRN scaling
* :func:`apply_rotary_pos_emb`
* :class:`Qwen3MoeAttention`  -- GQA with decoupled ``head_dim`` and per-head QK-norm
* :class:`Qwen3MoeMLP`        -- dense SwiGLU, used for ``mlp_only_layers``
* :func:`build_causal_mask`, :func:`build_document_mask` for pretraining packing

Shapes follow ``(batch, seq, hidden)`` at the module boundary and
``(batch, heads, seq, head_dim)`` inside attention.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3MoeConfig

__all__ = [
    "ACT2FN",
    "Qwen3MoeAttention",
    "Qwen3MoeMLP",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "build_causal_mask",
    "build_document_mask",
    "mask_value",
    "repeat_kv",
    "rotate_half",
    "supports_gqa_sdpa",
]


@functools.lru_cache(maxsize=1)
def supports_gqa_sdpa() -> bool:
    """Whether ``F.scaled_dot_product_attention`` accepts ``enable_gqa`` (torch>=2.5)."""
    try:
        probe = torch.zeros(1, 2, 1, 4)
        F.scaled_dot_product_attention(
            torch.zeros(1, 4, 1, 4), probe, probe, enable_gqa=True
        )
        return True
    except (TypeError, RuntimeError):
        return False


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


ACT2FN = {
    "silu": _silu,
    "swish": _silu,
    "gelu": _gelu,
    "gelu_pytorch_tanh": _gelu,
}


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (T5 / Llama style).

    The variance is always accumulated in fp32 and the normalised activations are cast
    back to the input dtype *before* multiplying by the learnable scale, which is what
    the reference Qwen3 implementation does.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: ``[x1, x2] -> [-x2, x1]``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(b, kv_heads, s, d)`` to ``(b, kv_heads * n_rep, s, d)``."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q``/``k`` of shape ``(batch, heads, seq, head_dim)``.

    ``cos``/``sin`` have shape ``(batch, seq, head_dim)`` or ``(seq, head_dim)``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with optional YaRN scaling.

    ``inv_freq`` is stored in fp32 as a non-persistent buffer so it follows the module
    across ``.to(device)`` / dtype conversions without ending up in checkpoints.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_scaling = dict(config.rope_scaling) if config.rope_scaling else None
        self.attention_scaling = 1.0

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ helpers
    def _compute_inv_freq(self) -> torch.Tensor:
        """Return ``(head_dim // 2,)`` fp32 inverse frequencies."""
        dim = self.head_dim
        base = float(self.rope_theta)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        scaling = self.rope_scaling
        if not scaling:
            return inv_freq

        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type == "linear":
            factor = float(scaling["factor"])
            return inv_freq / factor
        if rope_type != "yarn":
            return inv_freq

        # --- YaRN (Peng et al., 2023) -------------------------------------
        factor = float(scaling["factor"])
        # Qwen3 checkpoints store the pre-extension length; fall back to the current max
        # so an un-extended config still behaves like plain RoPE (factor=1).
        original_max = float(
            scaling.get("original_max_position_embeddings", self.max_position_embeddings)
        )
        beta_fast = float(scaling.get("beta_fast", 32))
        beta_slow = float(scaling.get("beta_slow", 1))

        # Correction range in "original" position units. ``beta_fast`` yields the small
        # index (high frequency, left untouched), ``beta_slow`` the large index.
        def _correction_dim(num_rotations: float) -> float:
            return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
                2 * math.log(base)
            )

        low = max(math.floor(_correction_dim(beta_fast)), 0)
        high = min(math.ceil(_correction_dim(beta_slow)), dim // 2 - 1)
        if high <= low:
            high = low + 1

        # ramp(i) == 1 -> extrapolate (keep the original frequency)
        # ramp(i) == 0 -> interpolate (divide by ``factor``)
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0.0, 1.0
        )
        inv_freq_interp = inv_freq / factor
        inv_freq = inv_freq * ramp + inv_freq_interp * (1.0 - ramp)

        # Magnitude scaling applied to cos/sin after rotation. The transformers default
        # is 0.1 * ln(factor) + 1.0; an explicit config value wins.
        attention_factor = scaling.get("attention_factor")
        if attention_factor is None:
            attention_factor = 0.1 * math.log(factor) + 1.0
        self.attention_scaling = float(attention_factor)
        return inv_freq

    # ------------------------------------------------------------------ forward
    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``cos``/``sin`` of shape ``(batch, seq, head_dim)`` in ``x.dtype``."""
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        # (batch, seq, head_dim/2) -> broadcast with fp32 positions for accuracy
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        rope_type = self.rope_type
        return (
            f"head_dim={self.head_dim}, theta={self.rope_theta}, "
            f"max_pos={self.max_position_embeddings}, type={rope_type}"
        )

    @property
    def rope_type(self) -> str:
        if not self.rope_scaling:
            return "default"
        rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        return "yarn" if rope_type == "yarn" else rope_type


class Qwen3MoeAttention(nn.Module):
    """Grouped-query attention with per-head QK RMSNorm.

    Deliberate deviations from a vanilla Llama attention block:

    * ``head_dim`` is an explicit config field (128), *not* ``hidden_size // heads`` (64),
      so the Q/K/V projections are wider than the hidden size.
    * ``q_norm`` / ``k_norm`` normalise each head independently, sharing one 128-element
      weight vector across every head of every layer.
    * ``v_proj`` has no bias and no norm.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.attn_impl = config.attn_impl

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run attention.

        Args:
            hidden_states: ``(batch, seq, hidden_size)``
            position_embeddings: ``(cos, sin)`` each ``(batch, seq, head_dim)``
            attention_mask: additive float mask broadcastable to
                ``(batch, heads, seq, kv_seq)``; ``0`` = keep, ``-inf`` = mask.
            past_key_value: optional ``(k, v)`` cache of shape
                ``(batch, kv_heads, past_seq, head_dim)`` for incremental decoding.

        Returns:
            ``(attn_output (b, s, hidden), attn_weights or None, present_kv or None)``.
            ``attn_weights`` is ``(b, heads, seq, kv_seq)`` in the activation dtype and is
            only materialised on the ``attn_impl="eager"`` path --- the SDPA path never
            builds it, which is what saves both memory and bandwidth during pretraining.
        """
        batch, seq_len, _ = hidden_states.shape
        head_shape = (batch, seq_len, -1, self.head_dim)

        # --- project, reshape per head, QK-norm (fp32 internally) -------------
        query_states = self.q_norm(self.q_proj(hidden_states).view(head_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(head_shape))
        value_states = self.v_proj(hidden_states).view(head_shape)

        # (batch, heads, seq, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        if cos.dim() == 2:  # (seq, head_dim) -> add batch dim
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        present_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        if use_cache:
            present_kv = (key_states, value_states)

        attn_weights: Optional[torch.Tensor] = None
        use_gqa = self.num_key_value_groups > 1
        if self.attn_impl == "sdpa" and (not use_gqa or supports_gqa_sdpa()):
            # F.scaled_dot_product_attention applies ``scale`` and, on torch>=2.5, expands
            # the KV heads internally when ``enable_gqa=True``. ``enable_gqa`` is only
            # passed when needed because torch<2.5 rejects the kwarg outright.
            #
            # ``is_causal`` is only safe when there is no cache: SDPA aligns its causal
            # mask to the top-left of the (query, key) pair, which is wrong as soon as the
            # query is shorter than the cached key sequence.
            sdpa_kwargs = {
                "attn_mask": attention_mask,
                "dropout_p": self.attention_dropout if self.training else 0.0,
                "scale": self.scaling,
                "is_causal": attention_mask is None and past_key_value is None,
            }
            if use_gqa:
                sdpa_kwargs["enable_gqa"] = True
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, **sdpa_kwargs
            )
        else:
            key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
            scores = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            elif past_key_value is None:
                # No explicit mask means "plain causal" (the SDPA path handles this with
                # ``is_causal``); the eager path must build the mask or pretraining would
                # silently attend to future tokens.
                causal = torch.full(
                    (seq_len, seq_len),
                    torch.finfo(scores.dtype).min,
                    dtype=scores.dtype,
                    device=scores.device,
                ).triu(1)
                scores = scores + causal
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            scores = F.dropout(scores, p=self.attention_dropout, training=self.training)
            attn_weights = scores
            attn_output = torch.matmul(scores, value_states_rep)

        # (b, s, heads, head_dim) -> (b, s, heads * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, present_kv


class Qwen3MoeMLP(nn.Module):
    """Dense SwiGLU feed-forward, used for ``mlp_only_layers`` (and for ablations)."""

    def __init__(self, config: Qwen3MoeConfig, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        inter = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_size = hidden_size
        self.intermediate_size = inter
        self.gate_proj = nn.Linear(hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- masks
def _mask_value(dtype: torch.dtype) -> float:
    """The additive "masked out" fill value for ``dtype``.

    ``finfo(dtype).min`` is correct for fp32/fp64 and also for bf16 (which shares fp32's
    8-bit exponent, so its minimum is exactly ``-3.3895e38``). fp16's minimum is
    ``-65504``, which is finite and equally fine for softmax. Callers must avoid *adding*
    two such values (that would produce ``-inf``), which is why
    :meth:`Qwen3MoeModel._build_mask` clamps the sum before the cast.
    """
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).min)
    return -1.0e9


def mask_value(dtype: torch.dtype) -> float:
    """Public alias of the dtype-safe masked-fill value."""
    return _mask_value(dtype)


def build_causal_mask(
    seq_len: int,
    past_len: int = 0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Additive causal mask of shape ``(1, 1, seq_len, past_len + seq_len)``."""
    total = past_len + seq_len
    # Query i (absolute position past_len + i) may attend to key j <= past_len + i.
    rows = torch.arange(seq_len, device=device).unsqueeze(1) + past_len
    cols = torch.arange(total, device=device).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros((1, 1, seq_len, total), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, _mask_value(dtype))
    return mask

def build_document_mask(
    document_ids: torch.LongTensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-diagonal causal mask for packed pretraining documents.

    Args:
        document_ids: ``(batch, seq)`` int tensor where equal ids form one document.

    Returns:
        ``(batch, 1, seq, seq)`` additive mask: causal *and* same-document only.
    """
    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be (batch, seq), got {tuple(document_ids.shape)}")
    batch, seq_len = document_ids.shape
    same_doc = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)  # (b, s, s)
    rows = torch.arange(seq_len, device=document_ids.device).unsqueeze(1)
    cols = torch.arange(seq_len, device=document_ids.device).unsqueeze(0)
    causal = cols <= rows
    allowed = (same_doc & causal).unsqueeze(1)  # (b, 1, s, s)
    mask = torch.zeros((batch, 1, seq_len, seq_len), dtype=dtype, device=document_ids.device)
    mask.masked_fill_(~allowed, _mask_value(dtype))
    return mask
