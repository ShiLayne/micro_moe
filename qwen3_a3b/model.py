"""Qwen3-30B-A3B decoder assembled for from-scratch pretraining.

Layout::

    48 x DecoderLayer
        input_layernorm            RMSNorm(2048)
        self_attn                  GQA 32/4, head_dim 128, per-head q/k norm
        residual
        post_attention_layernorm   RMSNorm(2048)
        mlp                        SparseMoeBlock (128 experts, top-8)
        residual
    final norm                     RMSNorm(2048)
    lm_head                        2048 -> 151936, untied

The forward pass returns per-layer router logits so the training loop can build the
load-balancing / z-loss terms and utilisation metrics without a second pass. Loss
computation is inline (``chunked_linear_cross_entropy`` + ``compute_moe_losses``) so a
single ``model(...)`` call yields ``loss``, ``ce_loss``, ``aux_loss`` and ``z_loss``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3MoeConfig, human_count
from .layers import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    RMSNorm,
    RotaryEmbedding,
    build_causal_mask,
    build_document_mask,
    mask_value,
)
from .loss import chunked_linear_cross_entropy, compute_moe_losses
from .moe import SparseMoeBlock
from .vision import VisionBlock

__all__ = [
    "CausalLMOutput",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "apply_fsdp2",
    "build_param_groups",
]

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class CausalLMOutput:
    """Structured output of :class:`Qwen3MoeForCausalLM`."""

    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    router_logits: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[KVCache] = None
    num_valid_tokens: Optional[torch.Tensor] = None


class Qwen3MoeDecoderLayer(nn.Module):
    """Pre-norm transformer block with an attention + (MoE) FFN residual branch."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_moe = config.is_moe_layer(layer_idx)
        if self.is_moe:
            self.mlp: nn.Module = SparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.use_checkpoint = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Returns ``(hidden, router_logits_or_None, present_kv_or_None)``."""
        caching = use_cache or past_key_value is not None
        # Activation recomputation only makes sense on the training path, where there is no
        # KV cache to thread through the checkpoint boundary. Dense and MoE layers use
        # different targets so the checkpoint output is always a pure tensor (or a tuple of
        # tensors), never a ``None``.
        if self.use_checkpoint and self.training and not caching:
            if self.is_moe:
                out, router_logits = torch.utils.checkpoint.checkpoint(
                    _moe_layer_forward_no_cache,
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    use_reentrant=False,
                )
                return out, router_logits, None
            out = torch.utils.checkpoint.checkpoint(
                _dense_layer_forward_no_cache,
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                use_reentrant=False,
            )
            return out, None, None

        return self._forward_impl(
            hidden_states, position_embeddings, attention_mask, past_key_value, use_cache
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Attention + FFN residuals; returns ``(hidden, router_logits, present_kv)``."""
        residual = hidden_states
        attn_out, _, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, router_logits = self.mlp(normed)  # type: ignore[misc]
        else:
            mlp_out, router_logits = self.mlp(normed), None
        return residual + mlp_out, router_logits, present


def _moe_layer_forward_no_cache(
    layer: "Qwen3MoeDecoderLayer",
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient-checkpoint target for a MoE layer: returns ``(hidden, router_logits)``.

    Defined at module scope so ``torch.utils.checkpoint`` does not have to capture a
    closure, and always returning two tensors so non-reentrant checkpointing has no
    ``None`` outputs to cope with.
    """
    out, router_logits, _ = layer._forward_impl(
        hidden_states, position_embeddings, attention_mask, None, False
    )
    assert router_logits is not None, "MoE layer must produce router logits"
    return out, router_logits


def _dense_layer_forward_no_cache(
    layer: "Qwen3MoeDecoderLayer",
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Gradient-checkpoint target for a dense layer: returns just the hidden states."""
    out, _, _ = layer._forward_impl(
        hidden_states, position_embeddings, attention_mask, None, False
    )
    return out


class Qwen3MoeModel(nn.Module):
    """Transformer trunk: embeddings, decoder layers, final norm."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

        # Optional vision tower. Kept on the trunk (not the ForCausalLM wrapper) so
        # ``model.model.vision_tower`` mirrors the Qwen-VL state-dict layout.
        if config.vision is not None:
            from .vision import Qwen3VisionModel, VisionAligner

            self.vision_tower: Optional[Qwen3VisionModel] = Qwen3VisionModel(config.vision)
            self.vision_aligner: Optional[VisionAligner] = VisionAligner(config)
        else:
            self.vision_tower = None
            self.vision_aligner = None

        self.gradient_checkpointing = False

    # ------------------------------------------------------------- checkpointing
    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable
        for layer in self.layers:
            layer.use_checkpoint = enable

    def enable_input_require_grads(self) -> None:
        """Make the embedding output require grad (needed with frozen embeddings)."""

        def _hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
            return output.requires_grad_(True)

        self.embed_tokens.register_forward_hook(_hook)

    # ---------------------------------------------------------------- multimodal
    def extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the ViT tower and the alignment block.

        Args:
            pixel_values: ``(num_images, 3, 224, 224)``.

        Returns:
            ``(num_images, num_patches, language_hidden)`` -- 196 tokens per image, already
            in the decoder's width.
        """
        if self.vision_tower is None or self.vision_aligner is None:
            raise RuntimeError(
                "this model has no vision tower; build it with a config that sets "
                "Qwen3MoeConfig.vision"
            )
        vision_features = self.vision_tower(pixel_values)
        return self.vision_aligner(vision_features)

    def splice_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embedding rows at ``image_token_id`` positions with image features.

        The placeholder count must match ``num_images * num_patches`` exactly. Placeholders
        are consumed in row-major order of ``input_ids``, which is what a chat template that
        emits ``<|vision_start|> + 196 x <|image_pad|> + <|vision_end|>`` produces.

        Raises:
            ValueError: on a placeholder/feature count mismatch, with the numbers in the
                message -- this is the most common multimodal wiring bug.
        """
        image_token_id = self.config.image_token_id
        if image_token_id is None:
            raise ValueError("config.image_token_id is not set, cannot place image features")

        features = self.extract_image_features(pixel_values)  # (n_img, n_patch, hidden)
        num_images, num_patches, hidden = features.shape

        mask = input_ids == image_token_id
        num_placeholders = int(mask.sum())
        if num_placeholders != num_images * num_patches:
            raise ValueError(
                f"found {num_placeholders} image placeholders (id={image_token_id}) but the "
                f"vision tower produced {num_images} x {num_patches} = "
                f"{num_images * num_patches} features. Each image occupies exactly "
                f"{num_patches} placeholder positions."
            )
        if hidden != inputs_embeds.shape[-1]:
            raise ValueError(
                f"aligned vision features are {hidden}-wide but the text embeddings are "
                f"{inputs_embeds.shape[-1]}-wide"
            )

        # Flatten in row-major (batch, seq) order so placeholders map to features in the same
        # order the images were passed.
        flat_features = features.reshape(-1, hidden).to(inputs_embeds.dtype)
        flat_embeds = inputs_embeds.reshape(-1, hidden).clone()
        flat_embeds[mask.reshape(-1)] = flat_features
        return flat_embeds.view(inputs_embeds.shape)

    # ------------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        output_router_logits: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[KVCache]]:
        """Run the trunk.

        Args:
            input_ids: ``(batch, seq)`` token ids (mutually exclusive with ``inputs_embeds``).
            pixel_values: ``(batch, 3, 224, 224)`` images. Required when ``input_ids``
                contains ``config.image_token_id`` placeholders; the vision tower's features
                overwrite the embedding at exactly those positions.
            position_ids: ``(batch, seq)``; defaults to ``0..seq-1``. Pass explicit values
                when packing documents so each document restarts at position 0.
            attention_mask: either a boolean ``(batch, seq)`` padding mask, or an
                additive float mask broadcastable to ``(batch, 1, seq, kv_seq)``.
            document_ids: ``(batch, seq)`` ids used to build a block-diagonal causal mask
                for packed pretraining sequences (overrides ``attention_mask``).
            use_cache: return per-layer ``(k, v)`` tensors for incremental decoding.

        Returns:
            ``(last_hidden_state, router_logits_per_moe_layer, present_kv)``
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids / inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # type: ignore[arg-type]

        # Multimodal splice: vision features overwrite the embedding rows that hold
        # ``image_token_id``. Doing it here (not as a concatenated prefix) keeps the
        # attention mask, position ids and KV cache identical to the text-only path.
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("pixel_values requires input_ids so image placeholders can be located")
            inputs_embeds = self.splice_image_features(
                input_ids, inputs_embeds, pixel_values
            )

        batch, seq_len = inputs_embeds.shape[:2]

        past_len = 0
        if past_key_values:
            past_len = past_key_values[0][0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch, -1)

        mask = self._build_mask(
            attention_mask, document_ids, inputs_embeds, seq_len, past_len
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_router_logits: List[torch.Tensor] = []
        present_kv: Optional[KVCache] = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, router_logits, layer_present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if output_router_logits and router_logits is not None:
                all_router_logits.append(router_logits)
            if present_kv is not None and layer_present is not None:
                present_kv.append(layer_present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_router_logits, present_kv

    @staticmethod
    def _build_mask(
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        seq_len: int,
        past_len: int,
    ) -> Optional[torch.Tensor]:
        """Turn the caller's mask description into an additive float mask.

        Returns ``None`` when the plain fused causal path can be used, otherwise a float
        tensor broadcastable to ``(batch, heads, seq, past_len + seq_len)`` in the model's
        activation dtype.
        """
        out_dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        neg = mask_value(torch.float32)

        def _finalize(mask: torch.Tensor) -> torch.Tensor:
            """Clamp to the destination dtype range, then cast.

            A fp32 ``finfo.min`` cast to fp16 becomes ``-inf``, and a full row of ``-inf``
            makes the softmax output NaN. Clamping first keeps every row finite.
            """
            return mask.clamp_min(mask_value(out_dtype)).to(out_dtype)

        if document_ids is not None:
            # Block-diagonal causal mask: packed documents cannot see each other.
            mask = build_document_mask(document_ids, dtype=torch.float32)
            if past_len:
                # Fully visible history columns for cached decoding.
                history = torch.zeros(
                    (mask.shape[0], 1, seq_len, past_len),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                mask = torch.cat([history, mask], dim=-1)
            return _finalize(mask)

        if attention_mask is None:
            if past_len == 0:
                return None  # let SDPA use its fused causal kernel
            return _finalize(
                build_causal_mask(seq_len, past_len, dtype=torch.float32, device=device)
            )

        if attention_mask.dtype == torch.bool:
            # (batch, seq) padding mask -> additive (batch, 1, seq, kv_seq)
            if attention_mask.dim() != 2:
                raise ValueError("boolean attention_mask must be (batch, seq)")
            total = past_len + seq_len
            if past_len and attention_mask.shape[1] == total:
                # Caller supplied a mask covering the whole KV sequence (incremental
                # decoding): use it verbatim.
                keep = attention_mask
            else:
                keep = attention_mask
                if past_len:
                    # Mask covers only the new tokens: history was already filtered when it
                    # entered the cache, so mark it fully visible.
                    keep = torch.cat(
                        [
                            torch.ones(
                                (keep.shape[0], past_len), dtype=torch.bool, device=keep.device
                            ),
                            keep,
                        ],
                        dim=1,
                    )
            if keep.shape[1] != total:
                raise ValueError(
                    f"attention_mask has {keep.shape[1]} positions but the KV sequence is "
                    f"{total} long ({past_len} cached + {seq_len} new)"
                )
            causal = build_causal_mask(
                seq_len, past_len, dtype=torch.float32, device=keep.device
            )  # (1, 1, seq, kv_seq)
            pad_mask = torch.zeros(
                (keep.shape[0], 1, seq_len, total), dtype=torch.float32, device=keep.device
            )
            pad_mask.masked_fill_(~keep[:, None, None, :], neg)
            # Adding two ``finfo.min`` values would overflow to -inf; clamp afterwards.
            return _finalize(pad_mask + causal)

        # Already additive: normalise to 4D and let SDPA broadcast.
        mask = attention_mask
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return _finalize(mask.to(torch.float32))


@dataclass
class _LossConfig:
    """Loss hyper-parameters that live outside the architecture config."""

    ignore_index: int = -100
    ce_chunk_size: int = 1024
    z_loss_coef: float = 0.0
    global_batch_aux: bool = True


class Qwen3MoeForCausalLM(nn.Module):
    """Trunk + untied LM head + pretraining loss wiring."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.extra_loss_config = _LossConfig(z_loss_coef=config.router_z_loss_coef)
        self._last_moe_metrics: Dict[str, torch.Tensor] = {}
        # Computed before ``apply`` so the language initialiser can skip vision submodules
        # with an O(1) id lookup instead of walking the tower for every module.
        self._vision_module_ids: Set[int] = self._collect_vision_module_ids()
        self.apply(self._init_module_weights)
        self._init_residual_outputs()
        # Vision must be initialised last: ``apply`` above walks every submodule with the
        # *language* recipe (std=0.02, RMSNorm/Linear assumptions) which is wrong for the
        # CLIP-style tower (LayerNorm biases, truncated-normal weights, a smaller qkv init).
        self._init_vision_weights()
        self.tie_weights()

    # ----------------------------------------------------------------- init
    def _init_module_weights(self, module: nn.Module) -> None:
        """From-scratch initialisation for the language model.

        ``nn.Module.apply`` visits children before parents, so the ``SparseMoeBlock``
        branch below runs after its ``gate`` / ``experts`` children were checked (and
        skipped, because ``TopKRouter`` and ``GroupedExperts`` hold bare parameters
        rather than ``nn.Linear`` sub-modules).

        Vision submodules are skipped here; they get the CLIP recipe in
        :meth:`_init_vision_weights`, which runs afterwards.
        """
        if id(module) in self._vision_module_ids:
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, SparseMoeBlock):
            module.gate.reset_parameters(std)
            module.experts.reset_parameters(
                std,
                self.config.use_depth_scaled_residual_init,
                self.config.num_hidden_layers,
            )

    def _collect_vision_module_ids(self) -> Set[int]:
        """Ids of every module owned by the vision tower / aligner (before ``apply``)."""
        ids = set()
        for root in (self.model.vision_tower, self.model.vision_aligner):
            if root is None:
                continue
            for sub in root.modules():
                ids.add(id(sub))
        return ids

    def _init_residual_outputs(self) -> None:
        """Shrink residual-writing dense projections for deep from-scratch training.

        ``o_proj`` and the dense FFN ``down_proj`` add directly into the residual stream;
        scaling their init by ``1/sqrt(2 * num_layers)`` keeps activation variance roughly
        constant across 48 pre-norm blocks. Expert ``down_proj`` matrices are handled
        inside :meth:`GroupedExperts.reset_parameters`.
        """
        if not self.config.use_depth_scaled_residual_init:
            return
        std = self.config.initializer_range / (2.0 * self.config.num_hidden_layers) ** 0.5
        for layer in self.model.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=std)
            if not layer.is_moe:
                nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=std)

    def _init_vision_weights(self) -> None:
        """Initialisation for the ViT tower and the alignment module.

        The two halves follow different conventions, matching their normalisation:

        * **ViT** -- CLIP recipe: weights ~ ``N(0, 0.02)`` truncated at 2 sigma, biases 0,
          LayerNorm weight 1 / bias 0, learned position table from the same distribution,
          ``qkv``/``out_proj``/``fc2`` scaled by ``1/sqrt(2 * depth)``.
        * **Aligner** -- decoder-adjacent recipe: same truncated normal, but RMSNorm has
          **weight only**, so there is no bias to zero.

        The alignment block runs at the language width and writes into the residual stream
        the decoder reads, so its output projections get the same depth scaling as ``o_proj``.
        """
        if self.model.vision_tower is None:
            return

        tower = self.model.vision_tower
        aligner = self.model.vision_aligner
        assert aligner is not None
        config = self.config.vision
        assert config is not None
        std = config.initializer_range

        def truncated_normal_(tensor: torch.Tensor, scale: float, cutoff: float = 2.0) -> None:
            with torch.no_grad():
                tensor.normal_(0.0, scale)
                # Resample the rare tails instead of clamping, which would create spikes.
                limit = cutoff * scale
                for _ in range(8):
                    bad = tensor.abs() > limit
                    if not bool(bad.any()):
                        break
                    tensor[bad] = torch.empty_like(tensor[bad]).normal_(0.0, scale)

        def init_attn_mlp(
            block: VisionBlock, residual_std: float, *, norm_has_bias: bool
        ) -> None:
            """Shared attention/MLP init; ``norm_has_bias`` distinguishes LN from RMSNorm."""
            truncated_normal_(block.attn.qkv.weight, residual_std)
            nn.init.zeros_(block.attn.qkv.bias)
            truncated_normal_(block.attn.out_proj.weight, residual_std)
            nn.init.zeros_(block.attn.out_proj.bias)
            truncated_normal_(block.mlp.fc1.weight, std)
            nn.init.zeros_(block.mlp.fc1.bias)
            truncated_normal_(block.mlp.fc2.weight, residual_std)
            nn.init.zeros_(block.mlp.fc2.bias)
            for norm in (block.norm1, block.norm2):
                nn.init.ones_(norm.weight)
                if norm_has_bias:
                    nn.init.zeros_(norm.bias)  # LayerNorm only; RMSNorm has no bias

        # --- ViT tower (LayerNorm + learned positions) --------------------------
        vit_residual_std = std / math.sqrt(2 * config.num_hidden_layers)
        truncated_normal_(tower.patch_embed.weight, std)
        nn.init.zeros_(tower.patch_embed.bias)
        truncated_normal_(tower.position_ids, std)
        for block in tower.blocks:
            init_attn_mlp(block, vit_residual_std, norm_has_bias=True)
        nn.init.ones_(tower.post_layernorm.weight)
        nn.init.zeros_(tower.post_layernorm.bias)

        # --- alignment (RMSNorm) ------------------------------------------------
        # The projection is the only path from vision into the decoder: keep it at full
        # std so image signal is not attenuated before it reaches the residual stream.
        truncated_normal_(aligner.proj.weight, std)
        nn.init.zeros_(aligner.proj.bias)
        align_residual_std = std / math.sqrt(2 * config.align_num_hidden_layers)
        for block in aligner.blocks:
            init_attn_mlp(block, align_residual_std, norm_has_bias=False)
        nn.init.ones_(aligner.norm.weight)

    # ----------------------------------------------------------------- helpers
    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        self.model.enable_gradient_checkpointing(enable)

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.model.embed_tokens.weight.numel()
            # With ``tie_word_embeddings`` the LM head *is* the embedding tensor, so it is
            # already excluded above; subtracting it twice would under-report the count.
            if not self.config.tie_word_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    def describe(self) -> str:
        breakdown = self.config.estimate_parameters()
        vision_params = self.config.estimate_vision_parameters()
        lines = [
            f"{self.config}",
            f"  language total : {human_count(self.num_parameters() - vision_params)}",
            f"  active/token   : {human_count(breakdown.active)}",
            f"  non-embedding  : {human_count(self.num_parameters(non_embedding=True) - vision_params)}",
        ]
        if self.config.vision is not None:
            lines.append(f"  vision+align   : {human_count(vision_params)}")
            lines.append(f"  grand total    : {human_count(self.num_parameters())}")
        return "\n".join(lines)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        output_router_logits: bool = True,
        return_logits: bool = True,
        ce_chunk_size: Optional[int] = None,
    ) -> CausalLMOutput:
        """Forward pass with optional pretraining losses.

        ``labels`` must already be the *shifted* targets aligned with ``input_ids``, i.e.
        ``labels[..., t]`` is the token that position ``t`` should predict. Use
        :func:`make_pretrain_batch` in ``train.py`` to build them.

        Pass ``pixel_values`` together with ``input_ids`` containing ``image_token_id``
        placeholders to run the multimodal path; omit it for text-only batches.
        """
        hidden_states, router_logits, present_kv = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
        )

        logits = None
        ce_loss = aux_loss = z_loss = None
        num_valid = None

        if labels is None:
            if return_logits:
                logits = self.lm_head(hidden_states)
        else:
            valid_mask = labels.reshape(-1) != self.extra_loss_config.ignore_index
            ce_loss, ce_z_loss, num_valid = chunked_linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                ignore_index=self.extra_loss_config.ignore_index,
                chunk_size=ce_chunk_size or self.extra_loss_config.ce_chunk_size,
                z_loss_coef=self.extra_loss_config.z_loss_coef,
            )
            moe_losses = compute_moe_losses(
                router_logits,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                aux_loss_coef=self.router_aux_loss_coef,
                z_loss_coef=self.extra_loss_config.z_loss_coef,
                mask=valid_mask,
                global_batch=self.extra_loss_config.global_batch_aux,
            )
            aux_loss = moe_losses.aux_loss
            z_loss = moe_losses.z_loss
            if self.extra_loss_config.z_loss_coef:
                # Vocabulary-level log-sum-exp term (PaLM), same coefficient as router z.
                z_loss = z_loss + ce_z_loss * self.extra_loss_config.z_loss_coef
            self._last_moe_metrics = dict(moe_losses.metrics)

        loss = None
        if ce_loss is not None:
            loss = ce_loss
            if aux_loss is not None:
                loss = loss + aux_loss
            if z_loss is not None:
                loss = loss + z_loss

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            z_loss=z_loss,
            router_logits=router_logits if output_router_logits else None,
            hidden_states=hidden_states,
            past_key_values=present_kv,
            num_valid_tokens=num_valid,
        )

    @property
    def last_moe_metrics(self) -> Dict[str, torch.Tensor]:
        """Routing health metrics (tensor scalars) from the most recent training step."""
        return getattr(self, "_last_moe_metrics", {})

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """Minimal greedy / sampled decoder loop with a KV cache.

        Intended for smoke tests after a checkpoint round-trip, not for production
        serving: no batching beyond a shared prompt length, no repetition penalties.

        ``pixel_values`` is consumed only on the first step, while the prompt (and therefore
        the image placeholders) is still in the sequence; later steps feed a single token
        and reuse the cache, so the vision tower runs exactly once per generation.
        """
        self.eval()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        generated = input_ids
        past: Optional[KVCache] = None
        # The boolean padding mask must cover the grown sequence, so it is extended for
        # every sampled token; passing it keeps SDPA from attending to pad positions.
        mask = attention_mask

        for _ in range(max_new_tokens):
            first_step = past is None
            step_ids = generated if first_step else generated[:, -1:]
            hidden_states, _, past = self.model(
                input_ids=step_ids,
                pixel_values=pixel_values if first_step else None,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                output_router_logits=False,
            )
            # Only the final position is needed to pick the next token.
            next_logits = self.lm_head(hidden_states[:, -1])  # (batch, vocab)

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits.float() / temperature
                if top_k is not None:
                    k = min(top_k, scaled.shape[-1])
                    kth = scaled.topk(k, dim=-1).values[..., -1:]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                next_token = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)
            if mask is not None:
                mask = torch.cat(
                    [mask, mask.new_ones((mask.shape[0], 1))], dim=-1
                )
            if eos is not None and bool((next_token == eos).all()):
                break
        return generated

    # ------------------------------------------------------------------ I/O
    def save_pretrained(self, save_directory: str, *, max_shard_bytes: int = 5 << 30) -> None:
        """Write ``config.json`` plus sharded safetensors (or a pickle fallback).

        Tensors are moved to host memory one at a time, so saving a 30B model never needs
        the whole state dict resident on the CPU at once. Tied weights are written once
        because safetensors rejects shared storage, and their binding is recorded in
        ``tied_weights.json``.
        """
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.config.to_hf_dict(), fh, indent=2)

        try:
            from safetensors.torch import save_file  # type: ignore
        except ImportError:  # pragma: no cover - safetensors is optional
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
            return

        # De-duplicate tied tensors by storage pointer, keeping the first name seen.
        state = self.state_dict()
        seen: Dict[int, str] = {}
        unique_keys: List[str] = []
        tied: Dict[str, str] = {}
        for key, tensor in state.items():
            ptr = tensor.data_ptr()
            if ptr in seen:
                tied[key] = seen[ptr]
                continue
            seen[ptr] = key
            unique_keys.append(key)

        # Pack into shards, flushing each one to disk as soon as it is full so only a single
        # shard is ever resident on the host.
        sizes = {key: state[key].numel() * state[key].element_size() for key in unique_keys}
        total_bytes = sum(sizes.values())

        # First pass computes the shard boundaries without touching tensor data.
        shard_of: Dict[str, int] = {}
        running = 0
        shard_idx = 0
        for key in unique_keys:
            if running and running + sizes[key] > max_shard_bytes:
                shard_idx += 1
                running = 0
            shard_of[key] = shard_idx
            running += sizes[key]
        total_shards = shard_idx + 1

        def _shard_name(idx: int) -> str:
            if total_shards == 1:
                return "model.safetensors"
            return f"model-{idx + 1:05d}-of-{total_shards:05d}.safetensors"

        index: Dict[str, str] = {}
        pending: Dict[str, torch.Tensor] = {}
        current = 0
        for key in unique_keys:
            if shard_of[key] != current:
                _flush_shard(pending, _shard_name(current), save_directory, save_file, index)
                pending = {}
                current = shard_of[key]
            pending[key] = state[key].detach().to("cpu", copy=True)
        _flush_shard(pending, _shard_name(current), save_directory, save_file, index)

        for key, target in tied.items():
            index[key] = index[target]

        if total_shards > 1:
            with open(
                os.path.join(save_directory, "model.safetensors.index.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(
                    {"metadata": {"total_size": total_bytes}, "weight_map": index}, fh, indent=2
                )

        if tied:
            with open(os.path.join(save_directory, "tied_weights.json"), "w", encoding="utf-8") as fh:
                json.dump({"tied_to": tied}, fh, indent=2)

    @classmethod
    def from_config(cls, config: Qwen3MoeConfig) -> "Qwen3MoeForCausalLM":
        model = cls(config)
        model.tie_weights()
        return model


# --------------------------------------------------------------------- training
def _flush_shard(
    shard: Dict[str, torch.Tensor],
    name: str,
    directory: str,
    save_file: object,
    index: Dict[str, str],
) -> None:
    """Write one safetensors shard and record its tensor -> file mapping."""
    if not shard:
        return
    import os as _os

    save_file(shard, _os.path.join(directory, name))  # type: ignore[operator]
    for key in shard:
        index[key] = name


def build_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> List[Dict[str, object]]:
    """Split parameters into decay / no-decay groups.

    No weight decay for: RMSNorm scales, embeddings, the LM head, the MoE router and any
    1-D bias-like tensor. Decay applies to every weight matrix (attention, dense MLP,
    expert matrices).
    """
    no_decay: List[str] = []
    decay: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm = param.dim() <= 1 or name.endswith(".bias")
        is_embed_or_head = "embed_tokens" in name or "lm_head" in name
        is_router = name.endswith("gate.weight")
        if is_norm or is_embed_or_head or is_router:
            no_decay.append(name)
        else:
            decay.append(name)

    named = dict(model.named_parameters())
    return [
        {
            "name": "decay",
            "params": [named[n] for n in decay],
            "weight_decay": weight_decay,
        },
        {
            "name": "no_decay",
            "params": [named[n] for n in no_decay],
            "weight_decay": 0.0,
        },
    ]


def apply_fsdp2(
    model: nn.Module,
    mesh: "torch.distributed.device_mesh.DeviceMesh",
    *,
    mp_policy: Optional[object] = None,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Wrap ``model`` with FSDP2 (``fully_shard``), one shard group per decoder layer.

    Each decoder layer becomes its own FSDP group, so only the current layer's parameters
    are all-gathered, and every expert tensor is sharded along dim 0 (the expert axis).
    Embedding and ``lm_head`` stay in the root group: they are needed at both ends of the
    step and re-sharding them costs more than it saves.
    """
    from torch.distributed.fsdp import fully_shard  # local: torch>=2.4 only

    kwargs: Dict[str, object] = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
    }
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy

    for layer in model.model.layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)
    return model
