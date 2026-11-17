"""Configuration for a from-scratch ``Qwen3-30B-A3B`` (``qwen3_moe``) model.

The numbers below mirror the official ``Qwen/Qwen3-30B-A3B-Base`` checkpoint, so a model
built from these presets has exactly the same shape as the released one. Analytic counts
produced by :meth:`Qwen3MoeConfig.estimate_parameters` (they match the published figures
up to rounding):

    total parameters      : 30.53B   (30,532,122,624)
    non-embedding params  : 29.91B   (29,909,792,768)
    activated params/token:  3.35B   ( 3,353,032,704)

Embedding and the untied LM head are 311.2M each (2.0% of the total each); the active
count above includes the embedding lookup and the LM head, as the Qwen3 report does.

Everything here is pure Python (no torch import) so the same config object can drive
static parameter accounting, checkpoint plumbing and data-pipeline sizing without
allocating a single byte of device memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

ExpertsImpl = Literal["eager", "grouped_mm"]
AttnImpl = Literal["sdpa", "eager"]

QWEN3_30B_A3B_VOCAB_SIZE = 151936


def human_count(n: int) -> str:
    """Format an integer parameter count as e.g. ``30.53B``."""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


@dataclass
class ParamBreakdown:
    """Static parameter accounting for the architecture described by a config.

    All fields are plain integers so this object is trivially serialisable.
    """

    embedding: int
    lm_head: int
    attention_per_layer: int
    norm_per_layer: int
    final_norm: int
    dense_mlp_per_layer: int
    router_per_layer: int
    experts_per_layer_total: int
    experts_per_layer_active: int
    num_layers: int
    num_moe_layers: int

    @property
    def num_dense_layers(self) -> int:
        return self.num_layers - self.num_moe_layers

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.final_norm
            + self.num_layers * (self.attention_per_layer + self.norm_per_layer)
            + self.num_dense_layers * self.dense_mlp_per_layer
            + self.num_moe_layers * (self.router_per_layer + self.experts_per_layer_total)
        )

    @property
    def active(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.final_norm
            + self.num_layers * (self.attention_per_layer + self.norm_per_layer)
            + self.num_dense_layers * self.dense_mlp_per_layer
            + self.num_moe_layers * (self.router_per_layer + self.experts_per_layer_active)
        )

    @property
    def non_embedding(self) -> int:
        return self.total - self.embedding - self.lm_head

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "active": self.active,
            "non_embedding": self.non_embedding,
            "embedding": self.embedding,
            "lm_head": self.lm_head,
            "attention_per_layer": self.attention_per_layer,
            "norm_per_layer": self.norm_per_layer,
            "final_norm": self.final_norm,
            "dense_mlp_per_layer": self.dense_mlp_per_layer,
            "router_per_layer": self.router_per_layer,
            "experts_per_layer_total": self.experts_per_layer_total,
            "experts_per_layer_active": self.experts_per_layer_active,
            "num_layers": self.num_layers,
            "num_moe_layers": self.num_moe_layers,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"total={human_count(self.total)} active={human_count(self.active)} "
            f"non_embedding={human_count(self.non_embedding)} "
            f"(layers={self.num_layers}, moe_layers={self.num_moe_layers})"
        )


@dataclass
class VisionConfig:
    """Hyper-parameters of the ViT tower plus its alignment module.

    Defaults describe a canonical **ViT-S/16**: 12 blocks, width 384, 6 heads, 16x16
    patches on 224x224 inputs, i.e. 196 patch tokens per image and ~21.7M parameters.

    Only ``transformers``-free pieces are used here (``torch``), and the field names follow
    ``CLIPVisionConfig`` where they overlap so a checkpoint is easy to map.
    """

    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    # The alignment module is *one* transformer block (plus a projection). Kept as a field
    # so the depth can be raised later without touching code; the default is the spec.
    align_num_hidden_layers: int = 1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError("vision hidden_size and num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "vision hidden_size must be divisible by num_attention_heads, got "
                f"{self.hidden_size} and {self.num_attention_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"vision head_dim must be even, got {self.head_dim}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, got "
                f"{self.image_size} and {self.patch_size}"
            )
        if self.align_num_hidden_layers < 1:
            raise ValueError("align_num_hidden_layers must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def grid_size(self) -> int:
        """Patches per spatial side (224 / 16 == 14)."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Patch tokens per image (14 * 14 == 196)."""
        return self.grid_size**2

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size**2

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "VisionConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in known})

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"VisionConfig(vit-s/{self.patch_size}: layers={self.num_hidden_layers}, "
            f"hidden={self.hidden_size}, heads={self.num_attention_heads}, "
            f"patches={self.num_patches}, align_blocks={self.align_num_hidden_layers})"
        )


@dataclass
class Qwen3MoeConfig:
    """Architecture hyper-parameters of a Qwen3 MoE style decoder.

    Field names intentionally match ``transformers.Qwen3MoeConfig`` so that a config
    can be dumped to / loaded from a Hugging Face ``config.json`` unchanged.
    """

    # --- vocabulary / backbone -------------------------------------------------
    vocab_size: int = QWEN3_30B_A3B_VOCAB_SIZE
    hidden_size: int = 2048
    intermediate_size: int = 6144  # dense SwiGLU hidden size (unused when every layer is MoE)
    num_hidden_layers: int = 48

    # --- attention -------------------------------------------------------------
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    # explicit head_dim, deliberately decoupled from hidden_size // num_attention_heads
    # (2048 // 32 == 64, but Qwen3 uses 128 -> q_proj is 2048x4096).
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None

    # --- normalisation / activations -------------------------------------------
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # --- positional encoding ---------------------------------------------------
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # --- mixture of experts ----------------------------------------------------
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    # Run the router matmul + softmax in fp32 regardless of the model dtype. bf16 logits
    # quantise a 128-way softmax badly during the first few thousand steps.
    router_fp32: bool = True

    # --- training-time regularisation -----------------------------------------
    router_aux_loss_coef: float = 1e-3
    router_z_loss_coef: float = 0.0

    # --- initialisation --------------------------------------------------------
    initializer_range: float = 0.02
    # Divide the std of residual-branch output projections (o_proj / down_proj) by
    # sqrt(2 * num_hidden_layers). Recommended for from-scratch pretraining of deep
    # stacks; set to False for bit-exact Hugging Face initialisation parity.
    use_depth_scaled_residual_init: bool = True

    # --- execution knobs (not part of the HF checkpoint) -----------------------
    moe_experts_impl: ExpertsImpl = "eager"
    attn_impl: AttnImpl = "sdpa"

    # --- misc / special tokens -------------------------------------------------
    tie_word_embeddings: bool = False
    use_cache: bool = True
    pad_token_id: Optional[int] = 151643
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645

    # --- vision tower (optional) ----------------------------------------------
    # ``None`` builds a text-only model. The image placeholder/start/end ids must be valid
    # indices of the *shared* embedding table, since image features are scattered into the
    # text embedding sequence at the placeholder positions.
    vision: Optional[VisionConfig] = None
    image_token_id: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None

    # ------------------------------------------------------------------ checks
    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, got "
                f"{self.num_attention_heads} and {self.num_key_value_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.num_experts < 0:
            raise ValueError("num_experts must be >= 0")
        if self.num_experts > 0:
            if not 0 < self.num_experts_per_tok <= self.num_experts:
                raise ValueError(
                    "num_experts_per_tok must be in (0, num_experts], got "
                    f"{self.num_experts_per_tok} and {self.num_experts}"
                )
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive for MoE models")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if any(i < 0 or i >= self.num_hidden_layers for i in self.mlp_only_layers):
            raise ValueError("mlp_only_layers entries must be valid layer indices")
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
            if rope_type not in ("default", "yarn", "linear", "dynamic"):
                raise ValueError(f"unsupported rope_scaling type: {rope_type}")
            if rope_type == "yarn" and self.rope_scaling.get("factor", 1.0) < 1.0:
                raise ValueError("YaRN factor must be >= 1.0")
        if self.moe_experts_impl not in ("eager", "grouped_mm"):
            raise ValueError(f"unknown moe_experts_impl: {self.moe_experts_impl}")
        if self.attn_impl not in ("sdpa", "eager"):
            raise ValueError(f"unknown attn_impl: {self.attn_impl}")
        if self.vision is not None:
            if self.image_token_id is None:
                raise ValueError(
                    "image_token_id is required when a vision tower is configured: image "
                    "features replace the embeddings at those positions"
                )
            for name in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
                value = getattr(self, name)
                if value is not None and not 0 <= value < self.vocab_size:
                    raise ValueError(
                        f"{name}={value} is outside the vocabulary [0, {self.vocab_size})"
                    )

    # ------------------------------------------------------------------ layout
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` uses a sparse MoE block (mirrors HF's rule)."""
        if self.num_experts <= 0:
            return False
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @property
    def num_moe_layers(self) -> int:
        return sum(1 for i in range(self.num_hidden_layers) if self.is_moe_layer(i))

    @property
    def num_dense_layers(self) -> int:
        return self.num_hidden_layers - self.num_moe_layers

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def rope_type(self) -> str:
        if self.rope_scaling is None:
            return "default"
        return self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))

    def estimate_vision_parameters(self) -> int:
        """Parameter count of the ViT tower + alignment block (0 when text-only).

        Counted separately from :meth:`estimate_parameters` because the vision tower is a
        fixed-cost add-on: it does not scale with language depth, so folding it into the
        language breakdown would make the per-layer numbers misleading.

        The two halves deliberately normalise differently, which changes their counts:

        * **ViT**: LayerNorm (weight **and** bias) + a learned position table, i.e. the CLIP
          convention, so it stays compatible with pretrained vision encoders.
        * **Aligner**: RMSNorm (weight only, no bias) at the language width, matching the
          decoder whose residual stream it feeds.

        Both use a standard 2-matrix GELU MLP with biases (not SwiGLU).
        """

        def vit_block(d: int, inter: int) -> int:
            """Pre-LayerNorm block: QKV+out, 2 LayerNorms (w+b), 2-matrix MLP."""
            attention = 3 * d * d + 3 * d + d * d + d
            norms = 2 * (d + d)  # weight + bias, twice
            mlp = d * inter + inter + inter * d + d
            return attention + norms + mlp

        def align_block(d: int, inter: int) -> int:
            """Pre-RMSNorm block: QKV+out, 2 RMSNorms (weight only), 2-matrix MLP."""
            attention = 3 * d * d + 3 * d + d * d + d
            norms = 2 * d  # weight only, no bias
            mlp = d * inter + inter + inter * d + d
            return attention + norms + mlp

        v = self.vision
        if v is None:
            return 0

        d = v.hidden_size
        patch_embed = v.patch_dim * d + d  # Conv2d weight + bias
        position_embed = v.num_patches * d  # learned absolute positions
        vit = (
            patch_embed
            + position_embed
            + v.num_hidden_layers * vit_block(d, v.intermediate_size)
            + 2 * d  # final LayerNorm weight + bias
        )

        h = self.hidden_size
        align = (
            d * h + h  # vision width -> language width, biased
            + v.align_num_hidden_layers * align_block(h, 4 * h)
            + h  # final RMSNorm weight only
        )
        return vit + align

    # ------------------------------------------------------- parameter counting
    def estimate_parameters(self) -> ParamBreakdown:
        """Analytically count total / activated parameters without building the model."""
        h = self.hidden_size
        head_dim = self.head_dim
        q_dim = self.q_proj_dim
        kv_dim = self.kv_proj_dim
        bias = int(self.attention_bias)

        attention_per_layer = (
            h * q_dim + bias * q_dim  # q_proj
            + h * kv_dim + bias * kv_dim  # k_proj
            + h * kv_dim + bias * kv_dim  # v_proj
            + q_dim * h + bias * h  # o_proj
            + head_dim  # q_norm
            + head_dim  # k_norm
        )
        norm_per_layer = 2 * h  # input_layernorm + post_attention_layernorm
        dense_mlp_per_layer = 3 * h * self.intermediate_size
        router_per_layer = h * self.num_experts if self.num_experts > 0 else 0
        expert_unit = 3 * h * self.moe_intermediate_size
        experts_per_layer_total = self.num_experts * expert_unit
        experts_per_layer_active = self.num_experts_per_tok * expert_unit

        embedding = self.vocab_size * h
        lm_head = 0 if self.tie_word_embeddings else h * self.vocab_size

        return ParamBreakdown(
            embedding=embedding,
            lm_head=lm_head,
            attention_per_layer=attention_per_layer,
            norm_per_layer=norm_per_layer,
            final_norm=h,
            dense_mlp_per_layer=dense_mlp_per_layer,
            router_per_layer=router_per_layer,
            experts_per_layer_total=experts_per_layer_total,
            experts_per_layer_active=experts_per_layer_active,
            num_layers=self.num_hidden_layers,
            num_moe_layers=self.num_moe_layers,
        )

    # ----------------------------------------------------------- (de)serialise
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Qwen3MoeConfig":
        """Build a config from a (Hugging Face style) dict, ignoring unknown keys.

        Accepts the vision tower under either spelling: ``vision`` (the dataclass field) or
        ``vision_config`` (the nested key that ``to_hf_dict`` and Qwen-VL configs emit).
        """
        # Resolve the alias *before* filtering: ``vision_config`` is not a dataclass field,
        # so a later rename would be discarded by the unknown-key filter.
        raw = dict(values)
        if "vision_config" in raw and "vision" not in raw:
            raw["vision"] = raw.pop("vision_config")
        else:
            raw.pop("vision_config", None)

        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in raw.items() if k in known}
        if "rope_scaling" in payload and isinstance(payload["rope_scaling"], dict):
            payload["rope_scaling"] = dict(payload["rope_scaling"])
        if "mlp_only_layers" in payload and payload["mlp_only_layers"] is None:
            payload["mlp_only_layers"] = []
        vision = payload.get("vision")
        if isinstance(vision, dict):
            payload["vision"] = VisionConfig.from_dict(vision)
        return cls(**payload)

    def to_hf_dict(self) -> Dict[str, Any]:
        """Emit a ``config.json`` payload compatible with ``transformers``."""
        payload: Dict[str, Any] = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "decoder_sparse_step": self.decoder_sparse_step,
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_only_layers": list(self.mlp_only_layers),
            "moe_intermediate_size": self.moe_intermediate_size,
            "norm_topk_prob": self.norm_topk_prob,
            "num_attention_heads": self.num_attention_heads,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_scaling": self.rope_scaling,
            "rope_theta": self.rope_theta,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "sliding_window": self.sliding_window,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
            "use_sliding_window": self.use_sliding_window,
            "vocab_size": self.vocab_size,
        }
        if self.vision is not None:
            # Mirrors the Qwen-VL layout: a nested vision_config plus the placeholder ids
            # at the top level.
            payload["vision_config"] = self.vision.to_dict()
            payload["image_token_id"] = self.image_token_id
            payload["vision_start_token_id"] = self.vision_start_token_id
            payload["vision_end_token_id"] = self.vision_end_token_id
        return payload

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"Qwen3MoeConfig(layers={self.num_hidden_layers}, hidden={self.hidden_size}, "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, experts={self.num_experts}"
            f"x{self.num_experts_per_tok}, moe_inter={self.moe_intermediate_size})"
        )


# --------------------------------------------------------------------------- presets
def qwen3_a300m_vision_config(**overrides: Any) -> Qwen3MoeConfig:
    """A300M + ViT-S/16 tower + one transformer-block alignment (Qwen-VL style).

    The image tokens are *not* appended to the sequence: the vision tower's 196 patch
    features are written into the text embedding sequence at the positions holding
    ``image_token_id``, which is how Qwen-VL family models splice images in. That keeps the
    language model's token budget unchanged by images and lets the same decoder handle both
    modalities.

    Sizes (measured, see ``estimate_parameters`` / ``estimate_vision_parameters``):
        language : 1.29B total / 295.99M active
        vision   : ~21.7M (ViT-S) + alignment block at the language width
    """
    values: Dict[str, Any] = dict(
        vision=VisionConfig(),          # ViT-S/16 defaults
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    values.update(overrides)
    return qwen3_a300m_config(**values)


def qwen3_30b_a3b_config(**overrides: Any) -> Qwen3MoeConfig:
    """Exact ``Qwen/Qwen3-30B-A3B-Base`` architecture (30.53B total / 3.35B active)."""
    values: Dict[str, Any] = dict(
        vocab_size=QWEN3_30B_A3B_VOCAB_SIZE,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        router_aux_loss_coef=1e-3,
        router_z_loss_coef=0.0,
        use_depth_scaled_residual_init=True,
        moe_experts_impl="grouped_mm",
    )
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def qwen3_a300m_config(**overrides: Any) -> Qwen3MoeConfig:
    """Architecture-preserving shrink of 30B-A3B to ~1.29B total / ~296M active.

    Mirrors ``configs/qwen3-a300m.json``. Ratios are preserved against the 30B model:
    ``q_proj_dim/hidden = 2.0``, ``moe_inter/hidden = 0.375``, ``top_k/num_experts =
    0.0625``, 4 KV heads, ``head_dim = 128``, all layers MoE.

    ``tie_word_embeddings=True`` is required, not cosmetic: at ``vocab_size=151936`` an
    untied head would make embedding + lm_head alone 311.2M parameters, already above the
    300M active budget.
    """
    values: Dict[str, Any] = dict(
        vocab_size=QWEN3_30B_A3B_VOCAB_SIZE,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=14,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=1_000_000.0,
        num_experts=64,
        num_experts_per_tok=4,
        moe_intermediate_size=384,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        router_aux_loss_coef=1e-3,
        router_z_loss_coef=1e-4,
        use_depth_scaled_residual_init=True,
        moe_experts_impl="grouped_mm",
        tie_word_embeddings=True,
    )
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def qwen3_2b_a600m_config(**overrides: Any) -> Qwen3MoeConfig:
    """Single-node experiment config: ~2.0B total / ~0.6B active, same MoE recipe.

    Sized by measurement, not by wishful naming: with the real 151936-token vocabulary the
    embedding plus untied LM head alone are 311M parameters, so a 3B-class active budget is
    the floor unless the vocabulary shrinks. Pass ``vocab_size=32768`` to get a ~350M-active
    variant for cheaper ablations.

    The ratios that shape routing dynamics are preserved (64 experts, top-8, expert width
    512 at hidden 1024), so experiments transfer to the 30B model, and the whole thing fits
    on a single 8 x 80GB node with FSDP2 at sequence length 4096.
    """
    values: Dict[str, Any] = dict(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=1_000_000.0,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        norm_topk_prob=True,
        router_aux_loss_coef=1e-3,
        router_z_loss_coef=1e-4,
        use_depth_scaled_residual_init=True,
        moe_experts_impl="grouped_mm",
    )
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def qwen3_tiny_debug_config(**overrides: Any) -> Qwen3MoeConfig:
    """Tiny config for shape / gradient / op-parity checks (CPU friendly)."""
    values: Dict[str, Any] = dict(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10_000.0,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        router_aux_loss_coef=1e-3,
        router_z_loss_coef=0.0,
        use_depth_scaled_residual_init=False,
        moe_experts_impl="eager",
        attn_impl="sdpa",
    )
    values.update(overrides)
    return Qwen3MoeConfig(**values)


CONFIG_PRESETS = {
    "qwen3-30b-a3b": qwen3_30b_a3b_config,
    "qwen3-a300m": qwen3_a300m_config,
    "qwen3-a300m-vision": qwen3_a300m_vision_config,
    "qwen3-2b-a600m": qwen3_2b_a600m_config,
    "qwen3-tiny-debug": qwen3_tiny_debug_config,
}


def build_config(name_or_path: str, **overrides: Any) -> Qwen3MoeConfig:
    """Resolve a preset name (or a ``config.json`` path) into a config object."""
    import json
    import os

    if name_or_path in CONFIG_PRESETS:
        return CONFIG_PRESETS[name_or_path](**overrides)
    if os.path.isfile(name_or_path):
        with open(name_or_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(overrides)
        return Qwen3MoeConfig.from_dict(payload)
    raise ValueError(
        f"unknown config '{name_or_path}', available presets: {sorted(CONFIG_PRESETS)}"
    )


__all__ = [
    "CONFIG_PRESETS",
    "ParamBreakdown",
    "Qwen3MoeConfig",
    "VisionConfig",
    "build_config",
    "human_count",
    "qwen3_2b_a600m_config",
    "qwen3_30b_a3b_config",
    "qwen3_a300m_config",
    "qwen3_a300m_vision_config",
    "qwen3_tiny_debug_config",
]
