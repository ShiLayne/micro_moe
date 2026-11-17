"""Minimal, dependency-light pretraining loop for Qwen3-30B-A3B.

Run with ``torchrun``::

    torchrun --standalone --nproc_per_node=8 train.py \
        --config qwen3-30b-a3b \
        --data /path/to/tokens.bin \
        --seq-len 4096 --micro-batch 2 --grad-accum 8 \
        --max-steps 100000 --lr 3e-4 --warmup-steps 2000 \
        --out-dir runs/qwen3-a3b --fsdp2 --gradient-checkpointing

What it does
------------
* Initialises ``torch.distributed`` (NCCL) from ``torchrun`` env vars, falling back to a
  single-process CPU/GPU run when launched directly.
* Wraps the model in FSDP2 (``fully_shard``) or plain DDP.
* Streams raw ``int32`` / ``int16`` token shards through ``torch.from_file`` (memory
  mapped) and builds shifted labels. :func:`make_pretrain_batch` also supports packed
  documents: pass ``document_boundaries`` to get restarting position ids and a
  block-diagonal causal mask so packed documents cannot attend to each other.
* Trains with bf16 autocast, gradient accumulation, AdamW (grouped decay), warmup +
  cosine schedule and global-norm clipping.
* Logs cross entropy, MoE aux loss, router z-loss and expert-load health; writes periodic
  checkpoints with config + optimizer state.

This file is a *skeleton*: intentionally readable rather than maximally optimised. Replace
:class:`TokenShardDataset` with your real data pipeline; everything else is ready to scale
by pointing it at more GPUs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP

from qwen3_a3b.config import Qwen3MoeConfig, build_config, human_count
from qwen3_a3b.loss import IGNORE_INDEX
from qwen3_a3b.model import Qwen3MoeForCausalLM, apply_fsdp2, build_param_groups


# --------------------------------------------------------------------------- args
@dataclass
class TrainArgs:
    config: str = "qwen3-tiny-debug"
    data: Optional[str] = None
    out_dir: str = "runs/qwen3-a3b"

    seq_len: int = 2048
    micro_batch: int = 1
    grad_accum: int = 1
    max_steps: int = 1000
    data_dtype: str = "int32"

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_schedule: str = "cosine"

    dtype: str = "bf16"
    fsdp2: bool = False
    gradient_checkpointing: bool = False
    global_batch_aux: bool = True
    ce_chunk_size: int = 1024

    log_every: int = 10
    ckpt_every: int = 1000
    seed: int = 1234
    dry_run: bool = False


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="From-scratch Qwen3-30B-A3B pretraining")
    defaults = TrainArgs()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        elif value is None:
            parser.add_argument(flag, type=str, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    namespace = parser.parse_args()
    return TrainArgs(**vars(namespace))


# ------------------------------------------------------------------------ runtime
def init_distributed() -> Tuple[int, int, int, torch.device]:
    """Return ``(rank, local_rank, world_size, device)``."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, local_rank, world_size, device

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return 0, 0, 1, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def autocast_context(device: torch.device, dtype: str):
    if dtype == "fp32" or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


# ------------------------------------------------------------------------- data
class TokenShardDataset:
    """Memory-mapped stream of pretraining tokens, cut into ``seq_len`` windows.

    The input is a flat binary file of little-endian token ids concatenated from your
    tokenizer output, read through ``torch.from_file`` so the corpus never has to fit in
    RAM. ``dtype`` selects the on-disk element width:

    * ``int32`` (default) -- required for the full 151936-token Qwen vocabulary
    * ``int16`` -- for small-vocab experiments (token ids must stay below 32768)

    Every ``seq_len`` tokens become one training sample, treated as a single document (plain
    causal mask). A production pipeline would instead pre-shuffle shards and emit real
    document boundaries (``make_pretrain_batch(document_boundaries=...)``); the interface
    here is identical, so swapping it out is a one-class change.
    """

    _DTYPES = {"int32": torch.int32, "int16": torch.int16}

    def __init__(
        self,
        path: str,
        seq_len: int,
        *,
        dtype: str = "int32",
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        pad_token_id: Optional[int] = None,
    ) -> None:
        if dtype not in self._DTYPES:
            raise ValueError(
                f"unsupported token dtype {dtype!r}; choose one of {sorted(self._DTYPES)}"
            )
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.dtype = self._DTYPES[dtype]
        self.tokens = torch.from_file(
            path, shared=False, size=self._numel(path, dtype), dtype=self.dtype
        )
        self.num_samples = self.tokens.numel() // seq_len
        if self.num_samples == 0:
            raise ValueError(f"{path} holds fewer than one sample of {seq_len} tokens")
        # Each rank walks a strided slice of the sample space so ranks never duplicate data.
        self.indices = torch.arange(rank, self.num_samples, world_size)
        self.generator = torch.Generator().manual_seed(seed + rank)
        self.epoch = 0

    @staticmethod
    def _numel(path: str, dtype: str) -> int:
        itemsize = 2 if dtype == "int16" else 4
        return os.path.getsize(path) // itemsize

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        order = self.indices[torch.randperm(self.indices.numel(), generator=self.generator)]
        for sample_idx in order.tolist():
            start = sample_idx * self.seq_len
            chunk = self.tokens[start : start + self.seq_len].to(torch.long)
            if chunk.numel() < self.seq_len:
                # Tail of the shard: skip instead of emitting a short (mis-shaped) sample.
                continue
            # Raw token shards carry no document metadata, so each window is one document
            # and the model falls back to a plain causal mask.
            yield make_pretrain_batch(chunk.unsqueeze(0), pad_token_id=self.pad_token_id)
        self.epoch += 1


def make_pretrain_batch(
    input_ids: torch.LongTensor,
    *,
    pad_token_id: Optional[int] = None,
    document_boundaries: Optional[torch.LongTensor] = None,
    image_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build ``(input_ids, labels, position_ids, document_ids)`` for next-token LM.

    Args:
        input_ids: ``(batch, seq)`` raw tokens.
        pad_token_id: if given, positions equal to this id are excluded from the loss.
        document_boundaries: optional ``(batch, seq)`` bool/int tensor that is 1 at the
            first token of every packed document. Position ids restart at 0 there and the
            block-diagonal attention mask follows the same seams.
        image_token_id: if given, every position holding this id is excluded from the loss
            -- image placeholders carry no token to predict, and training on them would
            teach the model to emit ``<|image_pad|>`` forever.

    Labels are the inputs shifted by one, i.e. ``labels[:, t] = input_ids[:, t+1]`` with the
    final position set to ``ignore_index``.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be (batch, seq)")
    batch, seq_len = input_ids.shape

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, :-1] = input_ids[:, 1:]
    if pad_token_id is not None:
        # Drop positions whose *target* is padding: the last real token before the pad run,
        # plus every position inside the run. Note this also drops legitimate occurrences of
        # ``pad_token_id`` if your tokenizer reuses a text token as pad, so pass ``None`` for
        # unpadded packed data.
        labels.masked_fill_(labels == pad_token_id, IGNORE_INDEX)
    if image_token_id is not None:
        # Also mask the position *predicting* an image placeholder for the same reason.
        labels.masked_fill_(input_ids == image_token_id, IGNORE_INDEX)
        labels.masked_fill_(labels == image_token_id, IGNORE_INDEX)

    position_ids = (
        torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    )

    if document_boundaries is None:
        # No packing: omit ``document_ids`` entirely so the model can use SDPA's fused
        # causal kernel instead of materialising a (batch, 1, seq, seq) block mask.
        document_ids = None
    else:
        starts = document_boundaries.to(torch.long)
        if starts.shape != input_ids.shape:
            raise ValueError("document_boundaries must have the same shape as input_ids")
        # Document index = number of starts seen so far, minus one. A document *includes*
        # its start token, hence the shift: starts=[1,0,0,1,0] -> ids=[0,0,0,1,1].
        # ``clamp_min(0)`` keeps the result sane when the first token is not a start.
        document_ids = (torch.cumsum(starts, dim=1) - 1).clamp_min(0)

        # Position ids restart at 0 inside every document. ``last_start`` is the index of the
        # most recent document start, obtained with a running max (no Python loop).
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        start_indices = positions.masked_fill(starts == 0, -1)
        last_start = torch.cummax(start_indices, dim=1).values.clamp_min(0)
        position_ids = positions - last_start

        # A document's last token must not be trained to predict the next document's first
        # token: that pair is unrelated and would inject pure label noise.
        next_is_start = torch.zeros_like(starts, dtype=torch.bool)
        next_is_start[:, :-1] = starts[:, 1:].to(torch.bool)
        labels.masked_fill_(next_is_start, IGNORE_INDEX)

    batch_dict: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
    }
    if document_ids is not None:
        batch_dict["document_ids"] = document_ids
    return batch_dict


def make_vision_pretrain_batch(
    input_ids: torch.LongTensor,
    pixel_values: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    num_patches: int,
    pad_token_id: Optional[int] = None,
    document_boundaries: Optional[torch.LongTensor] = None,
) -> Dict[str, torch.Tensor]:
    """Multimodal variant of :func:`make_pretrain_batch`.

    ``input_ids`` must already contain the placeholders: for each image, one
    ``vision_start_token_id``, ``num_patches`` copies of ``image_token_id`` and one
    ``vision_end_token_id``. The count must satisfy
    ``(#image_token_id positions) == (#images) * num_patches``; the model raises with both
    numbers if it does not, so this function only validates the cheap invariants.

    Returns the same dict as :func:`make_pretrain_batch` plus ``pixel_values``.
    """
    placeholders = int((input_ids == image_token_id).sum())
    num_images = pixel_values.shape[0]
    if placeholders != num_images * num_patches:
        raise ValueError(
            f"{placeholders} image placeholders vs {num_images} images x {num_patches} "
            "patches: every image needs exactly num_patches placeholder tokens"
        )
    if vision_start_token_id is not None:
        starts = int((input_ids == vision_start_token_id).sum())
        if starts != num_images:
            raise ValueError(
                f"{starts} vision_start tokens for {num_images} images"
            )
    if vision_end_token_id is not None:
        ends = int((input_ids == vision_end_token_id).sum())
        if ends != num_images:
            raise ValueError(f"{ends} vision_end tokens for {num_images} images")

    batch = make_pretrain_batch(
        input_ids,
        pad_token_id=pad_token_id,
        document_boundaries=document_boundaries,
        image_token_id=image_token_id,
    )
    batch["pixel_values"] = pixel_values
    return batch


def _cycle_epochs(
    shards: List["TokenShardDataset"],
) -> Iterator[Dict[str, torch.Tensor]]:
    """Stream batches from ``shards`` forever, reshuffling every epoch.

    A ``for epoch in itertools.count()`` loop over the dataset would stop after one pass;
    pretraining wants an endless stream, and re-seeding each epoch gives a fresh order.
    """
    epoch = 0
    while True:
        for shard in shards:
            shard.generator.manual_seed(shard.generator.initial_seed() + epoch)
            yield from shard
        epoch += 1


# ---------------------------------------------------------------------- schedule
def lr_lambda(step: int, args: TrainArgs) -> float:
    if step < args.warmup_steps:
        return (step + 1) / max(args.warmup_steps, 1)
    if args.lr_schedule == "constant":
        return 1.0
    progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def build_optimizer(model: nn.Module, args: TrainArgs) -> torch.optim.Optimizer:
    groups = build_param_groups(model, weight_decay=args.weight_decay)
    # FSDP2 / DDP wrap the same Parameter objects, so grouping by name still works.
    return torch.optim.AdamW(
        groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps
    )


# -------------------------------------------------------------------- checkpoint
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Qwen3MoeConfig,
    args: TrainArgs,
    rank: int,
) -> None:
    """Write a sharded-model checkpoint to ``args.out_dir``.

    FSDP2 keeps parameters as ``DTensor`` and ``full_tensor()`` is a *collective*: every
    rank has to call it, then rank 0 writes the file. Gathering on rank 0 only would hang
    the job.

    Note this is a simple full-state dump, adequate for the 30B model on a single node with
    the parameters already materialised. For multi-node runs, swap this for
    ``torch.distributed.checkpoint`` (sharded, resumable) or DTensor's
    ``save_state_dict`` --- the interface used by the training loop does not change.
    """
    # ``state_dict()`` is forwarded by DDP, and for FSDP2 the sharded DTensor values are
    # materialised inside :func:`_materialise`.
    state = model.state_dict()
    if is_main_process(rank):
        os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"step_{step:07d}.pt")
    # Both the parameters and the optimizer moments are DTensors under FSDP2, and
    # ``full_tensor()`` is a collective: *every* rank must materialise every tensor before
    # anyone writes to disk. Only rank 0 keeps the result.
    full_state = _materialise(state, keep=is_main_process(rank))
    full_optim = _materialise(optimizer.state_dict(), keep=is_main_process(rank))

    if not is_main_process(rank):
        return

    torch.save(
        {
            "step": step,
            "model": full_state,
            "optimizer": full_optim,
            "config": config.to_hf_dict(),
            "train_args": asdict(args),
        },
        path,
    )
    print(f"[ckpt] wrote {path}", flush=True)


def _materialise(value: object, *, keep: bool) -> object:
    """Recursively all-gather DTensors and move them to CPU.

    Called on **all** ranks (the gathers are collective). When ``keep`` is False the
    gathered tensors are dropped immediately so non-zero ranks pay only the communication
    cost, not the host memory.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.full_tensor() if hasattr(value, "full_tensor") else value
        return tensor.detach().cpu() if keep else None
    if isinstance(value, dict):
        return {key: _materialise(item, keep=keep) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialise(item, keep=keep) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialise(item, keep=keep) for item in value)
    return value


# ------------------------------------------------------------------------- train
def train(args: TrainArgs) -> None:
    rank, local_rank, world_size, device = init_distributed()
    torch.manual_seed(args.seed + rank)

    config = build_config(args.config)
    config.moe_experts_impl = "grouped_mm" if device.type == "cuda" else config.moe_experts_impl

    model = Qwen3MoeForCausalLM(config).to(device)
    model.extra_loss_config.global_batch_aux = args.global_batch_aux
    model.extra_loss_config.ce_chunk_size = args.ce_chunk_size
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing(True)

    # ``model`` may be a DDP or FSDP2 wrapper afterwards; all forward/state/optimizer paths
    # go through it so the wrappers' hooks stay active. Nothing needs the raw module.
    if is_main_process(rank):
        print(model.describe(), flush=True)
        breakdown = config.estimate_parameters()
        print(
            f"[params] total={human_count(breakdown.total)} "
            f"active/token={human_count(breakdown.active)} "
            f"layers={config.num_hidden_layers} moe_layers={config.num_moe_layers}",
            flush=True,
        )

    if args.fsdp2 and world_size > 1:
        mesh = init_device_mesh(
            "cuda" if device.type == "cuda" else "cpu", (world_size,)
        )
        model = apply_fsdp2(model, mesh)
    elif world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            # Every MoE layer's expert weights live in two dense 3D tensors, so a single
            # routed token is enough to produce a gradient for the whole tensor. No
            # parameter is ever unused, which lets DDP stay on the fast path.
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, args)
    )

    if args.data is None:
        if not args.dry_run:
            raise ValueError("--data is required unless --dry-run is set")
        # Synthetic tokens: exercises every shape without touching the filesystem.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        synthetic_batch = make_pretrain_batch(
            torch.randint(
                0,
                config.vocab_size,
                (args.micro_batch, args.seq_len),
                generator=generator,
            )
        )
        dataset: Iterator[Dict[str, torch.Tensor]] = iter(
            {k: v.clone() for k, v in synthetic_batch.items()} for _ in range(10**9)
        )
    else:
        shards = [
            TokenShardDataset(
                path,
                args.seq_len,
                dtype=args.data_dtype,
                seed=args.seed,
                rank=rank,
                world_size=world_size,
            )
            for path in args.data.split(",")
        ]
        dataset = _cycle_epochs(shards)

    model.train()
    step = 0
    micro = 0
    started = time.time()
    # Metrics stay as device tensors so the inner loop never synchronises with the host;
    # they are collapsed to floats only when a log line is actually emitted.
    running: Dict[str, List[torch.Tensor]] = {}

    def _record(key: str, value: torch.Tensor) -> None:
        running.setdefault(key, []).append(value.detach())

    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        micro += 1
        batch = next(dataset)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with autocast_context(device, args.dtype):
            # Call through ``model`` (which may be DDP / FSDP2): DDP prepares its
            # gradient-sync reducer inside its own ``forward`` and FSDP2 installs
            # pre-forward hooks on the wrapped module, so bypassing the wrapper would
            # silently disable both.
            out = model(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
                position_ids=batch["position_ids"],
                document_ids=batch.get("document_ids"),
                output_router_logits=True,
                ce_chunk_size=args.ce_chunk_size,
            )
            # Normalise by the accumulation factor so the gradient is a true batch mean.
            loss = out.loss / args.grad_accum  # type: ignore[operator]

        loss.backward()

        _record("loss", loss * args.grad_accum)
        if out.ce_loss is not None:
            _record("ce", out.ce_loss)
        if out.aux_loss is not None:
            _record("aux", out.aux_loss)
        # ``z_loss`` is a real tensor only when the z-loss coefficient is non-zero; checking
        # the config keeps this off the synchronisation path.
        if out.z_loss is not None and config.router_z_loss_coef:
            _record("z", out.z_loss)
        # The metrics live on the wrapper when DDP is used, so read them through the
        # wrapper rather than a captured inner module.
        for key, value in model.last_moe_metrics.items():
            _record(key, value)

        if micro % args.grad_accum != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], args.grad_clip
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            # One host sync per log line instead of one per micro-batch.
            means = {key: float(torch.stack(vals).mean()) for key, vals in running.items()}
            running = {}
            lr = optimizer.param_groups[0]["lr"]
            tokens = step * args.grad_accum * args.micro_batch * args.seq_len * world_size
            elapsed = max(time.time() - started, 1e-6)
            report = " ".join(f"{key}={value:.4f}" for key, value in means.items())
            print(
                f"[step {step}] {report} lr={lr:.2e} grad_norm={float(grad_norm):.3f} "
                f"tok/s={tokens / elapsed:,.0f}",
                flush=True,
            )

        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, step, config, args, rank)

    if dist.is_available() and dist.is_initialized():
        # Flush the final state before tearing the process group down.
        save_checkpoint(model, optimizer, step, config, args, rank)
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        # Build the model, run one synthetic forward/backward, then exit. Never writes
        # checkpoints and never touches the data path.
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        config = build_config(args.config)
        model = Qwen3MoeForCausalLM(config).to(device)
        model.train()

        seq = min(args.seq_len, 128)
        prompt = torch.randint(0, config.vocab_size, (args.micro_batch, seq), device=device)

        if config.vision is None:
            batch = make_pretrain_batch(prompt)
        else:
            # Multimodal smoke test: one image per batch row, its 196 patch placeholders
            # written into the first 196 text positions after a vision_start token.
            num_patches = config.vision.num_patches
            needed = num_patches + 2  # start + patches + end
            if seq < needed:
                raise SystemExit(
                    f"--seq-len must be at least {needed} for one image "
                    f"({num_patches} patches + start/end); got {seq}"
                )
            prompt = torch.randint(
                0, 1000, (args.micro_batch, seq), device=device
            )
            prompt[:, 0] = config.vision_start_token_id  # type: ignore[index]
            prompt[:, 1 : 1 + num_patches] = config.image_token_id  # type: ignore[index]
            prompt[:, 1 + num_patches] = config.vision_end_token_id  # type: ignore[index]
            images = torch.rand(
                args.micro_batch,
                config.vision.num_channels,
                config.vision.image_size,
                config.vision.image_size,
                device=device,
            )
            batch = make_vision_pretrain_batch(
                prompt,
                images,
                image_token_id=config.image_token_id,  # type: ignore[arg-type]
                vision_start_token_id=config.vision_start_token_id,
                vision_end_token_id=config.vision_end_token_id,
                num_patches=num_patches,
            )
            print(
                f"[dry-run] multimodal: {args.micro_batch} image(s) x {num_patches} patches "
                f"-> {num_patches * args.micro_batch} placeholder positions"
            )

        out = model(**batch)  # type: ignore[arg-type]
        out.loss.backward()  # type: ignore[union-attr]
        print(
            f"[dry-run] loss={float(out.loss):.4f} "  # type: ignore[arg-type]
            f"ce={float(out.ce_loss):.4f} "  # type: ignore[arg-type]
            f"aux={float(out.aux_loss):.4f}"  # type: ignore[arg-type]
        )
        print(f"[dry-run] {model.describe()}")
        _report_grad_health(model, config)
        return
    train(args)


def _report_grad_health(model: nn.Module, config: Qwen3MoeConfig) -> None:
    """Print how many parameters received a gradient in the dry-run backward pass.

    Every parameter should have a gradient: the MoE experts are stored in dense 3D tensors,
    so a single routed token yields a gradient for the whole expert tensor rather than
    leaving unused experts at ``grad=None``. With a vision tower, the vision and alignment
    parameters are checked separately because a silent placeholder mismatch is the most
    common way to end up with a frozen (or unused) tower.
    """
    vision_ids = {
        id(p) for m in (model.model.vision_tower, model.model.vision_aligner)
        if m is not None for p in m.parameters()
    }
    missing: List[str] = []
    missing_vision: List[str] = []
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total += 1
        if param.grad is None:
            missing.append(name)
            if id(param) in vision_ids:
                missing_vision.append(name)
    print(
        f"[dry-run] params with grad: {total - len(missing)}/{total}"
        + (f", missing (first 5): {missing[:5]}" if missing else "")
    )
    if config.vision is not None:
        vision_total = len([p for p in model.parameters() if id(p) in vision_ids])
        print(
            f"[dry-run] vision params with grad: "
            f"{vision_total - len(missing_vision)}/{vision_total}"
            + (
                f", e.g. {missing_vision[:5]}"
                if missing_vision
                else "  (tower is wired to the loss)"
            )
        )

    for layer in model.model.layers:
        if getattr(layer, "is_moe", False):
            gate_up_grad = layer.mlp.experts.gate_up_proj.grad  # type: ignore[union-attr]
            print(
                f"[dry-run] layer{layer.layer_idx} expert gate_up grad: "
                + ("None" if gate_up_grad is None else f"shape {tuple(gate_up_grad.shape)}")
            )
            break


if __name__ == "__main__":
    main()
