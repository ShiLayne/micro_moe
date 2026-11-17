"""Training losses for from-scratch Qwen3-30B-A3B pretraining.

Three pieces:

* :func:`chunked_linear_cross_entropy` -- next-token cross entropy over a 151936-way
  vocabulary without ever materialising the full ``(B, S, vocab)`` logits tensor.
* :func:`global_load_balancing_loss` -- Switch-Transformer style auxiliary loss whose
  assignment counts ``f_i`` are aggregated across data-parallel ranks (Qwen3's
  "global-batch load balancing"), while the differentiable probability term stays local.
* :func:`router_z_loss` -- optional log-sum-exp regulariser on the router logits,
  recommended when training a large MoE from scratch.

:func:`compute_moe_losses` wires all three together over the per-layer router logits and
fuses the cross-rank reduction into a **single** all-reduce per step (48 layers x 2
collectives per layer would otherwise dominate the step time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = [
    "IGNORE_INDEX",
    "MoeLossOutput",
    "all_reduce_sum",
    "assignment_counts",
    "chunked_linear_cross_entropy",
    "compute_moe_losses",
    "expert_load_metrics",
    "global_load_balancing_loss",
    "router_z_loss",
]

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- CE
def chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.LongTensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    chunk_size: int = 1024,
    z_loss_coef: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient ``lm_head`` + cross entropy.

    The full logits tensor for an 8k-token sequence is ``8192 * 151936 * 4 B ≈ 5 GB`` in
    fp32; chunking the vocabulary projection keeps the peak at
    ``chunk_size * vocab * 4 B`` (about 620 MB at the default chunk size).

    Args:
        hidden_states: ``(batch, seq, hidden)`` final hidden states. The caller is
            responsible for shifting: ``hidden_states[t]`` must predict ``labels[t]``.
        weight: ``(vocab, hidden)`` lm-head weight (may be tied to the embedding).
        labels: ``(batch, seq)`` target ids; ``ignore_index`` marks masked positions.
        chunk_size: token positions per matmul.
        z_loss_coef: if non-zero, also accumulate the token mean of
            ``logsumexp(logits)^2`` (PaLM's stability term). The returned value is
            unweighted; the caller applies the coefficient.

    Returns:
        ``(ce_loss, z_loss_term, num_valid_tokens)``: token-mean cross entropy, token-mean
        ``logsumexp^2`` and an int64 scalar with the number of supervised positions.
    """
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            f"shape mismatch: hidden_states {tuple(hidden_states.shape)} vs "
            f"labels {tuple(labels.shape)}"
        )

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    total_positions = flat_hidden.shape[0]
    device = flat_hidden.device

    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    z_sum = torch.zeros((), dtype=torch.float32, device=device)
    valid_total = torch.zeros((), dtype=torch.long, device=device)

    for start in range(0, total_positions, chunk_size):
        end = min(start + chunk_size, total_positions)
        chunk_labels = flat_labels[start:end]
        chunk_valid = chunk_labels != ignore_index
        valid_total = valid_total + chunk_valid.sum()

        logits = F.linear(flat_hidden[start:end], weight).float()
        # ``reduction="sum"`` already skips ignored positions and returns exactly 0.0 for a
        # chunk with no valid token, so no host-side "is this chunk empty?" test is needed.
        loss_sum = loss_sum + F.cross_entropy(
            logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
        )
        if z_loss_coef != 0.0:
            lse = torch.logsumexp(logits, dim=-1)
            z_sum = z_sum + (lse * chunk_valid.to(lse.dtype)).pow(2).sum()

    denom = valid_total.clamp_min(1).to(torch.float32)
    return loss_sum / denom, z_sum / denom, valid_total


# ------------------------------------------------------------------- MoE losses
def router_z_loss(
    router_logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean ``logsumexp(logits)^2`` over tokens (PaLM / ST-MoE stabiliser)."""
    lse = torch.logsumexp(router_logits.float(), dim=-1)
    if mask is not None:
        mask_f = mask.reshape(-1).to(lse.dtype)
        return (lse.pow(2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return lse.pow(2).mean()


def assignment_counts(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(counts (num_experts,), num_tokens)`` for one layer's top-k routing.

    ``counts`` is a detached fp32 histogram over expert ids; ``num_tokens`` is a scalar fp32
    tensor. Both live on the router-logits device. The aggregate path in
    :func:`compute_moe_losses` builds the histogram itself so that ``softmax`` / ``topk``
    only run once per layer; this helper is the standalone equivalent.
    """
    probs = F.softmax(router_logits.float(), dim=-1)
    top_indices = torch.topk(probs, top_k, dim=-1).indices
    if mask is not None:
        top_indices = top_indices[mask.reshape(-1).to(torch.bool)]
    counts = torch.bincount(top_indices.reshape(-1), minlength=num_experts).to(torch.float32)
    return counts.detach(), counts.new_tensor(float(top_indices.shape[0]))


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across data-parallel ranks when a process group is live."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    if tensor.device.type == "cuda" and dist.get_backend() != "nccl":
        reduced = tensor.cpu()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced.to(tensor.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def global_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    mask: Optional[torch.Tensor] = None,
    *,
    global_batch: bool = True,
) -> torch.Tensor:
    r"""Switch-Transformer load-balancing loss for a single layer.

    .. math::
        L_{aux} = N \sum_i f_i P_i

    ``f_i`` is the detached fraction of top-k assignments that landed on expert ``i`` and
    ``P_i`` is the mean router probability of expert ``i`` (differentiable).

    With ``global_batch=True`` the *counts* are summed across data-parallel ranks before the
    fraction is taken, so a rank that happens to see a narrow domain does not force its local
    experts towards a local-uniform distribution. ``P_i`` deliberately stays local:
    aggregating it would need a differentiable all-reduce for a negligible change in the
    gradient signal.

    This is the standalone single-layer entry point (useful for diagnostics). The training
    path uses :func:`compute_moe_losses`, which fuses all layers' collectives into one.

    Args:
        router_logits: ``(tokens, num_experts)`` fp32.
        mask: optional ``(tokens,)`` valid-token mask; masked rows are excluded from both
            terms so padding / ignored labels do not skew the balance.
    """
    probs = F.softmax(router_logits.float(), dim=-1)  # (T, E), differentiable
    top_indices = torch.topk(probs, top_k, dim=-1).indices
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool)
        probs = probs[valid]
        top_indices = top_indices[valid]
    if probs.shape[0] == 0:
        return router_logits.new_zeros(())

    counts = torch.bincount(
        top_indices.reshape(-1), minlength=num_experts
    ).to(torch.float32)
    totals = counts.new_tensor(float(probs.shape[0]))
    if global_batch:
        counts = all_reduce_sum(counts.detach())
        totals = all_reduce_sum(totals)

    assignment_fraction = (counts / totals.clamp_min(1.0)).detach()
    mean_prob = probs.mean(dim=0)  # differentiable, local
    return num_experts * torch.sum(assignment_fraction * mean_prob)


def expert_load_metrics(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    mask: Optional[torch.Tensor] = None,
    *,
    counts: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Scalar tensors describing routing health; call ``.item()`` only when logging.

    ``counts`` lets the caller reuse a histogram it already built (the aggregate path does
    this so a forward pass runs ``softmax``/``topk`` once per layer, not three times).
    """
    if counts is None:
        probs = F.softmax(router_logits.float(), dim=-1)
        _, top_indices = torch.topk(probs, top_k, dim=-1)
        if mask is not None:
            top_indices = top_indices[mask.reshape(-1).to(torch.bool)]
        if top_indices.numel() == 0:
            zero = router_logits.new_zeros(())
            return {"load_max": zero, "load_min": zero, "load_cv": zero}
        counts = torch.bincount(
            top_indices.reshape(-1), minlength=num_experts
        ).to(torch.float32)

    fraction = counts / counts.sum().clamp_min(1.0)
    uniform = 1.0 / num_experts
    return {
        # How far the hottest expert exceeds a perfectly uniform share (relative).
        "load_max": fraction.max() / uniform - 1.0,
        "load_min": 1.0 - fraction.min() / uniform,
        "load_cv": fraction.std(unbiased=False) / fraction.mean().clamp_min(1e-12),
    }


# -------------------------------------------------------------------- aggregate
@dataclass
class MoeLossOutput:
    """Coefficient-scaled auxiliary losses plus un-scaled metric tensors."""

    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


def compute_moe_losses(
    router_logits_per_layer: Sequence[torch.Tensor],
    num_experts: int,
    top_k: int,
    *,
    aux_loss_coef: float = 1e-3,
    z_loss_coef: float = 0.0,
    mask: Optional[torch.Tensor] = None,
    global_batch: bool = True,
) -> MoeLossOutput:
    """Average the load-balancing / z-loss terms over all MoE layers.

    The averages are already multiplied by their coefficients, so the caller can simply do
    ``loss = ce_loss + out.aux_loss + out.z_loss``.

    Two efficiency properties matter at 48 layers:

    * ``softmax`` and ``topk`` run **once per layer**; the same probabilities feed the aux
      loss, the z-loss and the utilisation metrics.
    * all per-layer assignment counts are stacked, so the cross-rank reduction is a single
      all-reduce per step instead of 48.
    """
    num_layers = len(router_logits_per_layer)
    if num_layers == 0:
        zero = torch.zeros((), dtype=torch.float32)
        return MoeLossOutput(aux_loss=zero, z_loss=zero, metrics={})

    device = router_logits_per_layer[0].device
    valid: Optional[torch.Tensor] = None
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool)
        if valid.shape[0] != router_logits_per_layer[0].shape[0]:
            raise ValueError(
                "mask must have one entry per routed token: "
                f"{valid.shape[0]} vs {router_logits_per_layer[0].shape[0]}"
            )

    # --- pass 1: probabilities, top-k, per-layer histograms -------------------
    counts_per_layer: List[torch.Tensor] = []
    probs_per_layer: List[torch.Tensor] = []
    lse_per_layer: List[torch.Tensor] = []
    for logits in router_logits_per_layer:
        logits_fp32 = logits.float()
        probs = F.softmax(logits_fp32, dim=-1)
        if valid is not None:
            probs = probs[valid]
        probs_per_layer.append(probs)
        counts_per_layer.append(
            torch.bincount(
                torch.topk(probs, top_k, dim=-1).indices.reshape(-1),
                minlength=num_experts,
            ).to(torch.float32)
        )
        if z_loss_coef != 0.0:
            # logsumexp over all experts, restricted to valid tokens afterwards.
            lse = torch.logsumexp(logits_fp32, dim=-1)
            lse_per_layer.append(lse if valid is None else lse[valid])

    counts_stacked = torch.stack(counts_per_layer, dim=0)  # (L, E)
    # Normalise by the number of *assignments* (tokens x top_k), not by the token count.
    # That is what makes ``sum_i f_i == 1`` and pins the loss to ~1.0 for a perfectly
    # balanced router --- the formulation used by DeepSeek-V3 and the Switch paper, and the
    # only one where ``router_aux_loss_coef`` means the same thing for any top_k.
    assignments_per_layer = counts_stacked.sum(dim=1)  # (L,)

    if global_batch:
        packed = torch.cat([counts_stacked.reshape(-1), assignments_per_layer], dim=0)
        packed = all_reduce_sum(packed)
        counts_stacked = packed[: counts_stacked.numel()].view_as(counts_stacked)
        assignments_per_layer = packed[counts_stacked.numel() :]

    # --- pass 2: reduce ------------------------------------------------------
    aux_total = torch.zeros((), device=device, dtype=torch.float32)
    z_total = torch.zeros((), device=device, dtype=torch.float32)
    load_max = torch.zeros((), device=device, dtype=torch.float32)
    load_cv = torch.zeros((), device=device, dtype=torch.float32)
    uniform = 1.0 / num_experts

    for layer_idx in range(num_layers):
        probs = probs_per_layer[layer_idx]
        counts = counts_stacked[layer_idx]
        if probs.shape[0] > 0:
            # f_i: fraction of top-k assignments that landed on expert i (detached)
            assignment_fraction = (
                counts / assignments_per_layer[layer_idx].clamp_min(1.0)
            ).detach()
            # P_i: mean router probability of expert i (differentiable, local)
            aux_total = aux_total + num_experts * torch.sum(
                assignment_fraction * probs.mean(dim=0)
            )

        if z_loss_coef != 0.0:
            lse = lse_per_layer[layer_idx]
            if lse.numel() > 0:
                z_total = z_total + lse.pow(2).mean()

        stats = expert_load_metrics(
            router_logits_per_layer[layer_idx],
            num_experts,
            top_k,
            counts=counts,
        )
        load_max = load_max + stats["load_max"]
        load_cv = load_cv + stats["load_cv"]

    inv_layers = 1.0 / num_layers
    aux_mean = aux_total * inv_layers
    z_mean = z_total * inv_layers

    metrics = {
        "moe/aux_loss_raw": aux_mean.detach(),
        "moe/load_max_violation": (load_max * inv_layers).detach(),
        "moe/load_cv": (load_cv * inv_layers).detach(),
    }
    if z_loss_coef != 0.0:
        metrics["moe/z_loss_raw"] = z_mean.detach()

    return MoeLossOutput(
        aux_loss=aux_mean * aux_loss_coef,
        z_loss=z_mean * z_loss_coef,
        metrics=metrics,
    )
