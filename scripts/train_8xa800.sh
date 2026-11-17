#!/usr/bin/env bash
#
# 8 x A800 (SM80, 80GB) launcher for the from-scratch Qwen3 pretraining loop.
#
# Default target is the A300M config (1.29B total / 296M active).
#
# Usage:
#   ./scripts/train_8xa800.sh /path/to/tokens.bin
#   MICRO_BATCH=16 GRAD_ACCUM=2 ./scripts/train_8xa800.sh /path/to/tokens.bin
#   ./scripts/train_8xa800.sh --make-data /tmp/smoke.bin 200   # 200MB random tokens
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/qwen3-a300m.json}"
NPROC="${NPROC:-8}"

# --- batch geometry -----------------------------------------------------------
# tokens per optimizer step = NPROC * MICRO_BATCH * SEQ_LEN * GRAD_ACCUM
# 8 * 8 * 4096 * 4 = 1,048,576 tokens/step
SEQ_LEN="${SEQ_LEN:-4096}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# --- optimisation -------------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-6e-4}"           # ~2x the 30B lr: smaller models tolerate larger steps
WARMUP="${WARMUP:-2000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DTYPE="${DTYPE:-bf16}"

# --- runtime ------------------------------------------------------------------
OUT_DIR="${OUT_DIR:-runs/qwen3-a300m-8xa800}"
LOG_EVERY="${LOG_EVERY:-10}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
GRAD_CKPT="${GRAD_CKPT:-0}"   # 1 = activation recomputation
DATA_DTYPE="${DATA_DTYPE:-int32}"

# FSDP2 shards parameters across ranks; DDP replicates them.
#
# For A300M (1.29B) DDP is the better choice on 8x80GB: the full replicated state is
#   bf16 params 2.6 GB + fp32 AdamW moments 10.3 GB + fp32 master 5.2 GB = ~18 GB/GPU,
# well inside 80 GB, and DDP avoids both the per-layer all-gather and the known
# FSDP2 friction with shared (tied) parameters -- which A300M uses.
#
# For 30B-A3B FSDP2 is mandatory: the replicated optimizer state alone would be ~244 GB.
FSDP2="${FSDP2:-0}"

# --------------------------------------------------------------------------- data
if [[ "${1:-}" == "--make-data" ]]; then
  OUT="${2:-/tmp/qwen3-smoke.bin}"
  MB="${3:-200}"
  python3 - "$OUT" "$MB" <<'PY'
import array, random, sys
path, megabytes = sys.argv[1], int(sys.argv[2])
total = megabytes * 1024 * 1024 // 4          # int32 elements
block = 1 << 20
rng = random.Random(0)
# Uniform ids in [0, 151936) so the sample exercises the real vocabulary range.
buf = array.array("i", (rng.randrange(0, 151936) for _ in range(block)))
written = 0
with open(path, "wb") as fh:
    while written < total:
        take = min(block, total - written)
        buf[:take].tofile(fh)
        written += take
sys.stderr.write(f"wrote {path}: {total} int32 tokens ({megabytes} MB)\n")
PY
  exit 0
fi

DATA="${1:-${DATA:-}}"
if [[ -z "$DATA" ]]; then
  echo "error: no token shard given." >&2
  echo "  usage: $0 /path/to/tokens.bin" >&2
  echo "  or generate a smoke shard: $0 --make-data /tmp/smoke.bin 200" >&2
  exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "error: data shard not found: $DATA" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "error: config not found: $CONFIG" >&2
  exit 1
fi

# ---------------------------------------------------------------------- preflight
# A800 is SM80. The fused grouped-GEMM kernel needs torch>=2.10 (F.grouped_mm, SM80+) or
# exactly 2.9 (torch._grouped_mm, SM80+); on older wheels it is SM90-only and MoE silently
# degrades to a per-expert Python loop. Surface that before burning GPU hours.
python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch is not installed in this interpreter")
if not torch.cuda.is_available():
    sys.exit("error: CUDA is not available")

n = torch.cuda.device_count()
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch {torch.__version__} | {n} x {name} | sm{cap[0]}{cap[1]} | {gib:.0f} GiB each")
if n < 8:
    print(f"warning: only {n} GPU(s) visible, NPROC will be capped accordingly")

parts = torch.__version__.split("+")[0].split(".")
vmaj_vmin = (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 0)

if hasattr(torch.nn.functional, "grouped_mm"):
    fused, why = cap >= (8, 0), "F.grouped_mm requires SM80+"
elif hasattr(torch, "_grouped_mm"):
    need = (9, 0) if vmaj_vmin < (9, 0) else (8, 0)
    fused, why = cap >= need, f"torch._grouped_mm requires SM{need[0]}{need[1]}+"
else:
    fused, why = False, "no grouped_mm API in this build"

print(f"fused MoE kernel : {'YES' if fused else 'NO -> per-expert loop'}  ({why})")
if not fused:
    print("  Numerics are identical either way; only MoE layer throughput changes.")
    print("  For a fused kernel on A800 use torch>=2.10, or exactly 2.9.")

if cap >= (8, 0) and cap < (9, 0):
    print("  SM80 detected: bf16 FlashAttention SDPA is supported (head_dim=128 is at the limit).")
PY

# ------------------------------------------------------------------ NCCL / threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Turn a hung collective into a raised error instead of an infinite wait.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Reduces allocator fragmentation; this job allocates many same-sized expert tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 8-GPU node with NVLink: keep P2P on the fabric.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
# No InfiniBand on a typical 8-GPU box; override if your node has a fabric.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

ARGS=(
  --config "$CONFIG"
  --data "$DATA"
  --seq-len "$SEQ_LEN"
  --micro-batch "$MICRO_BATCH"
  --grad-accum "$GRAD_ACCUM"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --warmup-steps "$WARMUP"
  --min-lr-ratio "$MIN_LR_RATIO"
  --grad-clip "$GRAD_CLIP"
  --weight-decay "$WEIGHT_DECAY"
  --dtype "$DTYPE"
  --out-dir "$OUT_DIR"
  --log-every "$LOG_EVERY"
  --ckpt-every "$CKPT_EVERY"
  --data-dtype "$DATA_DTYPE"
)
[[ "$GRAD_CKPT" == "1" ]] && ARGS+=(--gradient-checkpointing)
[[ "$FSDP2"     == "1" ]] && ARGS+=(--fsdp2)

echo "---------------------------------------------------------------"
echo "config        : $CONFIG"
echo "data          : $DATA"
echo "ngpu          : $NPROC"
echo "seq/micro/acc : $SEQ_LEN / $MICRO_BATCH / $GRAD_ACCUM"
echo "tokens/step   : $(( NPROC * MICRO_BATCH * SEQ_LEN * GRAD_ACCUM ))"
echo "lr            : $LR (warmup $WARMUP, min ratio $MIN_LR_RATIO)"
echo "fsdp2 / ckpt  : $FSDP2 / $GRAD_CKPT"
echo "out           : $OUT_DIR"
echo "---------------------------------------------------------------"

# --standalone already implies a local c10d rendezvous; do not also pass
# --rdzv_endpoint or you risk colliding with an existing listener on port 29500.
exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  train.py "${ARGS[@]}"

# -------------------------------------------------------------------------------
# Full Qwen3-30B-A3B on 8 x A800 80GB
# -------------------------------------------------------------------------------
# Per-GPU steady-state state under FSDP2 (everything sharded /8):
#   bf16 params        61 GB / 8 =  7.6 GB
#   bf16 grads         61 GB / 8 =  7.6 GB
#   AdamW moments     244 GB / 8 = 30.5 GB   (fp32 exp_avg + exp_avg_sq)
#   fp32 master       122 GB / 8 = 15.3 GB
#   ------------------------------------------
#   ~61 GB/GPU before activations, leaving ~19 GB for a single layer's all-gather
#   buffer plus recomputed activations. Workable, but only with activation
#   checkpointing and micro-batch 1. Validate the pipeline on A300M first.
#
#   python3 -c "import json;from qwen3_a3b.config import qwen3_30b_a3b_config as f;print(json.dumps(f().to_hf_dict(),indent=2))" > configs/qwen3-30b-a3b.json
#
#   CONFIG=configs/qwen3-30b-a3b.json \
#   SEQ_LEN=4096 MICRO_BATCH=1 GRAD_ACCUM=16 \
#   LR=2e-4 WARMUP=2000 GRAD_CKPT=1 FSDP2=1 \
#   OUT_DIR=runs/qwen3-30b-a3b-8xa800 \
#   ./scripts/train_8xa800.sh /path/to/tokens.bin
#
#   (8 * 1 * 4096 * 16 = 524,288 tokens/step)
