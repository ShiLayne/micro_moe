"""Mixture-of-Experts block for Qwen3-30B-A3B.

Design notes
------------
Qwen3's MoE is *fine grained and shared-expert free*: 128 routed SwiGLU experts per
layer, 8 selected per token, no always-on shared expert. The reference implementation
stores every expert as an independent ``nn.Linear`` set, which means 48 * 128 = 6144
sub-modules --- painful for FSDP sharding, optimizer traversal and CUDA graph capture.

This implementation instead keeps all experts of a layer in two contiguous 3D
parameters::

    gate_up_proj : (num_experts, 2 * moe_intermediate_size, hidden_size)
    down_proj    : (num_experts, hidden_size, moe_intermediate_size)

and dispatches tokens with one of two backends:

``eager``
    Sort ``(token, slot)`` pairs by expert id, then run one matmul per *active* expert.
    Exact reference semantics; works on CPU / MPS / any dtype.

``grouped_mm``
    Same permutation, but the per-expert matmuls are fused into a single
    ``grouped_mm`` call (the segment-GEMM kernel used for Qwen3 pretraining), with an
    automatic fallback to the ``eager`` path when the op is missing or the shapes/dtypes
    are unsupported (fp32 on CPU, bf16 rows not a multiple of 8, older torch builds).

Both backends are numerically equivalent up to floating point associativity.

One expert dimension detail worth stating explicitly: with 128 experts and top-8, an
average token touches ``8 / 128 = 6.25%`` of the sparse parameters, which is what makes a
30.5B model cost 3.35B active parameters per token.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3MoeConfig
from .layers import ACT2FN

__all__ = [
    "GroupedExperts",
    "SparseMoeBlock",
    "TopKRouter",
    "grouped_mm_available",
]


def _torch_major_minor() -> Tuple[int, int]:
    """``(major, minor)`` of the running torch build, ignoring any ``+cuXXX`` suffix."""
    parts = torch.__version__.split("+")[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


# Cached per-device capability verdict, so the check costs one dict lookup per layer
# instead of a CUDA query (or, worse, a thrown exception) per call.
_GROUPED_MM_SUPPORT: Dict[str, bool] = {}


def grouped_mm_available(device: Optional[torch.device] = None) -> bool:
    """Whether a **fused** grouped-GEMM kernel can run on ``device``.

    The private ``torch._grouped_mm`` needed SM90 (Hopper) until torch 2.9; the public
    ``F.grouped_mm`` accepts SM80+ (Ampere, i.e. A100 / A800 / A30 / RTX 30xx). Below the
    cutoff the op still *exists* but raises at call time, so probing with ``hasattr`` alone
    would throw on every MoE layer of every step on an A800.

    Args:
        device: CUDA device to test. ``None`` (or a non-CUDA device) is always
            unsupported, because the kernel is CUDA-only.

    Returns:
        ``True`` when the fused kernel is usable, ``False`` when callers should use the
        per-expert fallback.
    """
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return False

    key = str(device)
    cached = _GROUPED_MM_SUPPORT.get(key)
    if cached is not None:
        return cached

    ok = False
    if hasattr(F, "grouped_mm"):
        # torch >= 2.10: public API, SM80+.
        ok = torch.cuda.get_device_capability(device) >= (8, 0)
    elif hasattr(torch, "_grouped_mm"):
        # torch 2.9 relaxed the private op from SM90 to SM80; earlier builds are SM90-only.
        sm90_only = _torch_major_minor() < (9, 0)
        minimum = (9, 0) if sm90_only else (8, 0)
        ok = torch.cuda.get_device_capability(device) >= minimum

    _GROUPED_MM_SUPPORT[key] = ok
    return ok


def _grouped_mm(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    counts: torch.LongTensor,
) -> torch.Tensor:
    """Compute ``out[g] = tokens[g] @ weight[g].T`` for every expert group.

    Args:
        tokens: ``(total_rows, in_features)`` rows already sorted by expert id.
        weight: ``(num_experts, out_features, in_features)``.
        counts: ``(num_experts,)`` number of rows owned by each expert.

    Returns:
        ``(total_rows, out_features)`` in ``tokens.dtype``.

    ``grouped_mm`` takes ``mat_b`` as ``(num_groups, K, N)`` -- i.e. the transposed
    ``nn.Linear`` layout -- and ``offs`` as *inclusive* cumulative group ends of dtype
    int32. Anything unsupported (CPU, fp32, ragged bf16 alignment, pre-SM80 hardware,
    older torch) falls back to per-expert matmuls that produce the same numbers.
    """
    if tokens.shape[0] == 0:
        return tokens.new_empty((0, weight.shape[1]))

    # The fused kernel is bf16/fp16-only. fp32 (debug runs, or ``--dtype fp32``) must take
    # the reference path or the call would raise inside ATen.
    half_precision = tokens.dtype in (torch.bfloat16, torch.float16)

    if half_precision and grouped_mm_available(tokens.device):
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        mat_b = weight.transpose(-1, -2)  # (G, N, K) -> (G, K, N)
        try:
            grouped = getattr(F, "grouped_mm", None) or torch._grouped_mm  # type: ignore[attr-defined]
            return grouped(tokens, mat_b, offs=offs)
        except (TypeError, RuntimeError, NotImplementedError, AttributeError):
            # Defensive: a version skew, an unaligned expert group, or an unexpected
            # restriction. Disable the fast path for this device so the next call does not
            # pay the exception again, then fall through.
            _GROUPED_MM_SUPPORT[str(tokens.device)] = False

    return _grouped_mm_fallback(tokens, weight, counts)


def _grouped_mm_fallback(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    counts: torch.LongTensor,
) -> torch.Tensor:
    """Per-expert matmuls over contiguous slices (one host sync for all counts)."""
    out = tokens.new_empty((tokens.shape[0], weight.shape[1]))
    start = 0
    # A single ``.tolist()`` keeps this at one device sync per layer instead of one per
    # expert (128 syncs would serialise the whole step).
    for expert_idx, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        end = start + count
        out[start:end] = F.linear(tokens[start:end], weight[expert_idx])
        start = end
    return out


class TopKRouter(nn.Module):
    """Bias-free linear gate: fp32 softmax over all experts, top-k, optional renorm.

    Mirrors ``Qwen3MoeTopKRouter``: probabilities come from a softmax over *all* experts,
    then the top-k values are renormalised to sum to one (``norm_topk_prob=True`` for every
    released Qwen3 checkpoint).

    With ``router_fp32`` the gate matmul runs in fp32 with autocast disabled. That matters:
    under bf16 autocast ``F.linear`` would silently downcast fp32 inputs back to bf16, and a
    128-way softmax on bf16 logits is visibly noisier in the first few thousand steps.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.router_fp32 = config.router_fp32
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.reset_parameters(config.initializer_range)

    def reset_parameters(self, std: float = 0.02) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``(tokens, hidden)``.

        Returns:
            ``logits``  -- ``(tokens, num_experts)`` fp32 (for aux / z-loss & metrics)
            ``weights`` -- ``(tokens, top_k)`` in ``hidden_states.dtype``
            ``indices`` -- ``(tokens, top_k)`` int64 expert ids
        """
        if self.router_fp32:
            # ``autocast`` would re-cast fp32 operands back to bf16 inside ``F.linear``, so
            # the gate must run in an autocast-disabled region to actually produce fp32
            # logits. The cast of the 128x2048 router weight is a ~1 MB copy per layer.
            device_type = hidden_states.device.type
            if device_type in ("cuda", "cpu"):
                with torch.autocast(device_type=device_type, enabled=False):
                    logits = F.linear(hidden_states.float(), self.weight.float())
            else:
                # MPS / XLA do not participate in autocast; plain fp32 is enough.
                logits = F.linear(hidden_states.float(), self.weight.float())
        else:
            logits = F.linear(hidden_states, self.weight)

        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_values, top_indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return logits, top_values.to(hidden_states.dtype), top_indices


class GroupedExperts(nn.Module):
    """All routed experts of one layer packed into two 3D parameters."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.impl = config.moe_experts_impl
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )

    def reset_parameters(
        self,
        std: float = 0.02,
        depth_scaled_residual: bool = True,
        num_layers: int = 1,
    ) -> None:
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=std)
        # ``down_proj`` writes straight into the residual stream, so for from-scratch
        # pretraining of deep stacks we shrink it by sqrt(2 * num_layers) to keep the
        # residual variance roughly constant with depth.
        down_std = std / (2.0 * max(num_layers, 1)) ** 0.5 if depth_scaled_residual else std
        nn.init.normal_(self.down_proj, mean=0.0, std=down_std)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_indices: torch.LongTensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Args: ``(tokens, hidden)``, ``(tokens, k)`` ids, ``(tokens, k)`` weights."""
        num_tokens, top_k = top_indices.shape

        # --- permute: sort (token, slot) pairs so equal experts are contiguous ----
        flat_experts = top_indices.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        token_index = torch.div(order, top_k, rounding_mode="floor")
        tokens_sorted = hidden_states.index_select(0, token_index)
        counts = torch.bincount(flat_experts, minlength=self.num_experts)

        # --- expert compute ------------------------------------------------------
        if self.impl == "grouped_mm":
            gate_up = _grouped_mm(tokens_sorted, self.gate_up_proj, counts)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = _grouped_mm(self.act_fn(gate) * up, self.down_proj, counts)
        else:
            expert_out = self._forward_per_expert(tokens_sorted, counts)

        # --- unpermute: scale by router weight and scatter-add back --------------
        weights = top_weights.reshape(-1).index_select(0, order)
        weighted = expert_out * weights.unsqueeze(-1)
        out = expert_out.new_zeros((num_tokens, self.hidden_size))
        out.index_add_(0, token_index, weighted)
        # No "unused expert" patch is needed here: all experts of this layer share the two
        # 3D parameters above, so routing *any* token into the layer produces a gradient for
        # the full ``gate_up_proj`` / ``down_proj`` tensor. That is also why DDP needs no
        # ``find_unused_parameters`` and FSDP never sees a ``grad=None`` expert shard.
        return out

    def _forward_per_expert(
        self,
        tokens_sorted: torch.Tensor,
        counts: torch.LongTensor,
    ) -> torch.Tensor:
        """Reference path: one SwiGLU per active expert over its contiguous rows."""
        out = tokens_sorted.new_empty((tokens_sorted.shape[0], self.hidden_size))
        start = 0
        # A single ``.tolist()`` keeps this at one device sync per layer instead of one per
        # expert (128 syncs per layer would serialise the step).
        for expert_idx, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            end = start + count
            gate, up = F.linear(
                tokens_sorted[start:end], self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            out[start:end] = F.linear(
                self.act_fn(gate) * up, self.down_proj[expert_idx]
            )
            start = end
        return out


class SparseMoeBlock(nn.Module):
    """Router + grouped experts + weighted combine.

    Returns the block output *and* the raw router logits so the training loop can build
    the load-balancing / z-loss terms and expert-utilisation metrics in one pass.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.gate = TopKRouter(config)
        self.experts = GroupedExperts(config)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: ``(batch, seq, hidden)``.

        Returns:
            ``output``        -- ``(batch, seq, hidden)`` in the input dtype
            ``router_logits`` -- ``(batch * seq, num_experts)`` in fp32
        """
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)

        logits, top_weights, top_indices = self.gate(flat)
        out = self.experts(flat, top_indices, top_weights)
        return out.reshape(batch, seq_len, hidden), logits
