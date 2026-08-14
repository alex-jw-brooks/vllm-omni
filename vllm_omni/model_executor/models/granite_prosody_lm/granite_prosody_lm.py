# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM - Omni wrapper around GraniteMoeHybridForCausalLM.

Shared model class for both pipeline stages:
  text_norm  — AR text normalization via the LM head
  prosody    — NAR prosody prediction via dedicated NAR heads

Each stage loads a separate pre-merged checkpoint. The model_stage field
(set by the pipeline config) controls which code paths are active.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.granitemoehybrid import (
    GraniteMoeHybridAttentionDecoderLayer,
    GraniteMoeHybridForCausalLM,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.granite_prosody_lm.decode_utils import (
    compute_preamble_layout,
    greedy_ctc_decode,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.granite_prosody_lm import (
    GraniteProsodyLMConfig,
)

logger = init_logger(__name__)

_NUM_PROSODY_HEADS = 6
_N_PROSODY_DIMS = 5
_WORD_COL_SIZE = 6


class HybridCausalBidirAttention(nn.Module):
    """Drop-in replacement for vLLM Attention with hybrid causal/bidirectional.

    Positions [0, bidir_start) use causal attention; positions
    [bidir_start, seq_len) use bidirectional. Backend-agnostic: uses
    PyTorch F.scaled_dot_product_attention directly.

    Set the class-level ``_bidir_start`` before each model forward pass.
    """

    _bidir_start: int = 0

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        seq_len = query.shape[0]

        q = query.view(seq_len, self.num_heads, self.head_dim)
        k = key.view(seq_len, self.num_kv_heads, self.head_dim)
        v = value.view(seq_len, self.num_kv_heads, self.head_dim)

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # [1, heads, seq, dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        bidir_start = self._bidir_start

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if 0 < bidir_start < seq_len:
            min_val = torch.finfo(attn_weights.dtype).min
            causal_mask = torch.triu(
                torch.ones(bidir_start, seq_len, device=query.device),
                diagonal=1,
            ).bool()
            float_mask = torch.zeros(
                seq_len,
                seq_len,
                dtype=attn_weights.dtype,
                device=query.device,
            )
            float_mask[:bidir_start].masked_fill_(causal_mask, min_val)
            attn_weights = attn_weights + float_mask
        elif bidir_start >= seq_len:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device),
                diagonal=1,
            ).bool()
            attn_weights.masked_fill_(causal_mask, torch.finfo(attn_weights.dtype).min)
        # bidir_start == 0 → fully bidirectional, no mask needed
        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(q.dtype)
        out = torch.matmul(attn_weights, v)
        return out.squeeze(0).transpose(0, 1).contiguous().view(seq_len, -1)


def patch_hybrid_attention(model: nn.Module, vllm_config: VllmConfig) -> int:
    """Replace causal Attention with HybridCausalBidirAttention on all layers.

    Returns the number of layers patched.
    """
    fwd_ctx = vllm_config.compilation_config.static_forward_context
    patched = 0
    for layer in model.layers:
        if not isinstance(layer, GraniteMoeHybridAttentionDecoderLayer):
            raise NotImplementedError(
                f"NAR hybrid attention patch only supports attention layers, got {type(layer).__name__}."
            )
        sa = layer.self_attn
        old_prefix = sa.attn.layer_name
        del fwd_ctx[old_prefix]
        sa.attn = HybridCausalBidirAttention(
            sa.num_heads,
            sa.head_dim,
            sa.attention_multiplier,
            num_kv_heads=sa.num_key_value_heads,
        )
        patched += 1
    return patched


class NARHeads(nn.Module):
    """Dedicated prediction heads for NAR prosody decode.

    6 heads for prosody dimensions (dur, pitch×3, energy, silence).
    Emotion heads (arousal, valence) are not yet supported.
    """

    def __init__(self, hidden_size: int, num_codebook: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_codebook, bias=False) for _ in range(_NUM_PROSODY_HEADS)]
        )

    def forward_head(self, head_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        head = self.heads[head_idx]
        return head(hidden.to(head.weight.dtype))


class GraniteProsodyLMForConditionalGeneration(nn.Module):
    has_preprocess = True
    has_postprocess = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = cast(
            GraniteProsodyLMConfig,
            vllm_config.model_config.hf_config,
        )
        self.vllm_config = vllm_config
        self.model_stage = vllm_config.model_config.model_stage
        self.nar_mode = self.model_stage == "prosody"
        self.ctc_text_norm = self.model_stage == "text_norm"

        if self.model_stage not in ("text_norm", "prosody"):
            raise ValueError(f"Unknown model_stage={self.model_stage!r}. Expected 'text_norm' or 'prosody'.")

        # FIXME: wrapping ForCausalLM instead of the inner Model to avoid
        # reimplementing lm_head/logits_processor/load_weights, but this
        # triggers a spurious "SupportsLoRA.lora_manager not implemented"
        # warning. Consider wrapping GraniteMoeHybridModel directly.
        self.model = GraniteMoeHybridForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if self.model_stage == "prosody":
            num_codebook = self.config.num_end_id - self.config.num_start_id
            self.model.nar_heads = NARHeads(
                self.config.hidden_size,
                num_codebook,
            )

        if self.ctc_text_norm:
            head_size = self.config.ctc_head_vocab_size + (2 if self.config.ctc_editor_copy_op else 1)
            self.ctc_head = nn.Linear(
                self.config.hidden_size,
                head_size,
                bias=False,
            )

        if self.nar_mode:
            n = patch_hybrid_attention(self.model.model, vllm_config)
            logger.info(
                "Patched %d attention layers to hybrid causal/bidirectional (NAR mode)",
                n,
            )

        # LLM_GENERATION runner checks this to handle OmniOutput returns.
        self.have_multimodal_outputs = True

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        if self.ctc_text_norm:
            return self._ctc_text_norm_decode(
                input_ids,
                positions,
                **kwargs,
            )
        return self._nar_decode(
            input_ids,
            positions,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "compute_logits should not be called — both stages use dedicated heads "
            "(NAR heads for prosody, CTC head for text norm)."
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        return input_ids, input_embeds, {}

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> dict[str, object]:
        # Both stages return complete results from forward() as OmniOutput.
        return {}

    def _ctc_text_norm_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> OmniOutput:
        """NLE CTC text normalization: single forward pass + CTC greedy decode.

        Expects input_ids = raw_prefix + interleaved_edit, where the
        interleaved portion starts at the first blank_token_id. The prefix
        provides causal context (same as the reference use_raw_prefix=True
        path). Only the edit portion's logits are CTC-decoded.
        """
        cfg = self.config
        blank_positions = (input_ids == cfg.blank_token_id).nonzero(as_tuple=True)[0]
        if len(blank_positions) == 0:
            self.model.forward(
                input_ids,
                positions,
                inputs_embeds=None,
                **kwargs,
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "text_norm_tokens": [
                        torch.tensor([], dtype=torch.long),
                    ],
                },
            )
        prefix_len = int(blank_positions[0].item())

        hidden_states = self.model.forward(
            input_ids,
            positions,
            inputs_embeds=None,
            **kwargs,
        )
        logits = self.ctc_head(
            hidden_states.to(self.ctc_head.weight.dtype),
        ).float()

        edit_logits = logits[prefix_len:]
        edit_ids = input_ids[prefix_len:]

        copy_id = cfg.ctc_copy_index if cfg.ctc_editor_copy_op else None
        src_at_pos = None
        if copy_id is not None:
            src_at_pos = [(int(tid) if int(tid) != cfg.blank_token_id else None) for tid in edit_ids]
        decoded = greedy_ctc_decode(
            edit_logits,
            cfg.ctc_blank_index,
            copy_id=copy_id,
            src_at_pos=src_at_pos,
        )
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "text_norm_tokens": [
                    torch.tensor(decoded, dtype=torch.long),
                ],
            },
        )

    def _find_annot_start(self, input_ids: torch.Tensor) -> int | None:
        """Find the start of the annotation block in input_ids.

        The annotation block starts at the [SIL] token after [SEP_F0].
        Returns None if [SEP_F0] is not found (e.g. profiling with dummy inputs).
        """
        matches = (input_ids == self.config.sep_f0_token_id).nonzero(
            as_tuple=True,
        )[0]
        if len(matches) == 0:
            return None
        return int(matches[-1].item()) + 1

    def _compute_dim_index(
        self,
        annot_len: int,
        g_pre: int,
    ) -> list[int]:
        """Build per-position dimension index for the annotation block.

        Maps each position (within the annotation block, after [SIL]) to
        its NAR head index. The [SIL] prefix and [SEP_2] suffix get -1.
        """
        _, preamble_dims = compute_preamble_layout(
            self.config.emotion_control,
            list(self.config.nar_global_dims),
            self.config.compact_preamble,
        )
        dim_index = [-1]  # [SIL] prefix
        dim_index.extend(preamble_dims)
        n_words = (annot_len - 1 - g_pre - 1) // _WORD_COL_SIZE
        for _ in range(n_words):
            dim_index.extend(list(range(_WORD_COL_SIZE)))
        dim_index.append(-1)  # [SEP_2] suffix
        return dim_index

    def _nar_predict_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        annot_start: int,
        dim_index: list[int],
        tier_positions: set[int],
        tau: float,
        **kwargs,
    ) -> tuple[dict[int, int], dict[int, float]]:
        """Run one NAR prediction step.

        Returns (predictions, confidences) dicts mapping annotation-block
        position -> predicted token ID / confidence score.
        """
        hidden_states = self.model.forward(
            input_ids,
            positions,
            inputs_embeds=None,
            **kwargs,
        )

        mask_id = self.config.mask_token_id
        num_start = self.config.num_start_id
        nar_heads = self.model.nar_heads

        predictions: dict[int, int] = {}
        confidences: dict[int, float] = {}

        for pos in tier_positions:
            abs_pos = annot_start + pos
            if input_ids[abs_pos].item() != mask_id:
                continue
            dim = dim_index[pos]
            if dim < 0:
                continue

            logit = nar_heads.forward_head(dim, hidden_states[abs_pos])
            logit[-1] = float("-inf")

            if tau <= 0:
                pred_local = int(logit.argmax().item())
                conf = 1.0
            else:
                logit = logit / tau
                probs = F.softmax(logit, dim=-1)
                pred_local = int(torch.multinomial(probs, 1).item())
                conf = float(probs[pred_local].item())

            predictions[pos] = num_start + pred_local
            confidences[pos] = conf

        return predictions, confidences

    def _nar_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Iterative masked prediction for NAR prosody decode.

        Runs multiple transformer forward passes. At each iteration,
        NAR heads predict values for masked positions, and the most
        confident predictions are unmasked.

        Returns OmniOutput with prosody token IDs in multimodal_outputs.
        """
        cfg = self.config
        annot_start = self._find_annot_start(input_ids)
        if annot_start is None:
            # Profiling/warmup with dummy inputs — run backbone for memory
            # profiling but skip NAR decode logic.
            self.model.forward(
                input_ids,
                positions,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "prosody_tokens": [
                        torch.tensor([], dtype=torch.long),
                    ],
                },
            )
        annot_len = input_ids.shape[-1] - annot_start

        # Use fully bidirectional attention for NAR decode, matching
        # the reference's FA2 is_causal=False behavior.
        HybridCausalBidirAttention._bidir_start = 0

        g_pre, _ = compute_preamble_layout(
            cfg.emotion_control,
            list(cfg.nar_global_dims),
            cfg.compact_preamble,
        )
        dim_index = self._compute_dim_index(annot_len, g_pre)

        # Work on a mutable copy of input_ids
        working_ids = input_ids.clone()
        mask_id = cfg.mask_token_id

        tiers = cfg.nar_tiers
        iterations_per_tier = cfg.nar_iterations_per_tier

        if tiers:
            for tier_idx, tier in enumerate(tiers):
                t_iter = iterations_per_tier[tier_idx]
                logger.debug("NAR tier %d/%d: tier=%s, iterations=%d", tier_idx, len(tiers), tier, t_iter)

                global_dims = set(tier.get("global", []))
                word_dims = set(tier.get("word", []))
                tier_positions: set[int] = set()
                for i, d in enumerate(dim_index):
                    if d < 0:
                        continue
                    # Preamble positions are 1..g_pre (after SIL at 0)
                    if 1 <= i <= g_pre and d in global_dims:
                        tier_positions.add(i)
                    elif i > g_pre and d in word_dims:
                        tier_positions.add(i)

                if not tier_positions:
                    continue

                for pos in tier_positions:
                    working_ids[annot_start + pos] = mask_id

                logger.debug("NAR tier %d: %d mask positions", tier_idx, len(tier_positions))
                for t in range(t_iter):
                    is_last = t == t_iter - 1
                    tau = self._compute_tau(
                        cfg.nar_temperature,
                        t,
                        t_iter,
                    )
                    logger.debug("NAR tier %d iter %d/%d: tau=%.3f", tier_idx, t, t_iter, tau)

                    predictions, confidences = self._nar_predict_step(
                        working_ids,
                        positions,
                        annot_start,
                        dim_index,
                        tier_positions,
                        tau,
                        **kwargs,
                    )

                    if not predictions:
                        continue

                    if is_last:
                        unmask_positions = set(predictions.keys())
                    else:
                        unmask_positions = self._select_unmask_positions(
                            predictions,
                            confidences,
                            t,
                            t_iter,
                        )

                    for pos in unmask_positions:
                        working_ids[annot_start + pos] = predictions[pos]
        else:
            all_positions = {i for i, d in enumerate(dim_index) if d >= 0}
            for t in range(cfg.nar_iterations):
                is_last = t == cfg.nar_iterations - 1
                tau = self._compute_tau(
                    cfg.nar_temperature,
                    t,
                    cfg.nar_iterations,
                )
                predictions, confidences = self._nar_predict_step(
                    working_ids,
                    positions,
                    annot_start,
                    dim_index,
                    all_positions,
                    tau,
                    **kwargs,
                )
                if not predictions:
                    continue
                if is_last:
                    unmask_positions = set(predictions.keys())
                else:
                    unmask_positions = self._select_unmask_positions(
                        predictions,
                        confidences,
                        t,
                        cfg.nar_iterations,
                    )
                for pos in unmask_positions:
                    working_ids[annot_start + pos] = predictions[pos]

        final_annot = working_ids[annot_start:].tolist()
        logger.debug(
            "NAR decode complete: %d tokens, first5=%s, last5=%s", len(final_annot), final_annot[:5], final_annot[-5:]
        )
        annot_tensor = torch.tensor(final_annot, dtype=torch.long)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "output_token_ids": [annot_tensor],
                "prosody_tokens": [annot_tensor],
            },
        )

    @staticmethod
    def _compute_tau(
        base_temperature: float,
        t: int,
        t_iter: int,
    ) -> float:
        """Compute sampling temperature for NAR iteration t."""
        if t_iter <= 1:
            return base_temperature
        # Last iteration is greedy
        if t == t_iter - 1:
            return 0.0
        # Linear anneal
        return base_temperature * (1.0 - t / (t_iter - 1))

    @staticmethod
    def _select_unmask_positions(
        predictions: dict[int, int],
        confidences: dict[int, float],
        t: int,
        t_iter: int,
    ) -> set[int]:
        """Select positions to unmask using cosine schedule."""
        sorted_pos = sorted(
            predictions.keys(),
            key=lambda p: confidences[p],
            reverse=True,
        )
        n_total = len(sorted_pos)
        # Cosine schedule: ramps from few → all
        fraction = 1.0 - math.cos(math.pi * (t + 1) / t_iter)
        n_keep = max(1, math.ceil(n_total * fraction / 2.0))
        return set(sorted_pos[:n_keep])

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        model_weights = []
        loaded_keys: set[str] = set()
        for name, tensor in weights:
            if name.startswith("ctc_head."):
                param = dict(self.named_parameters())[name]
                param.data.copy_(tensor)
                loaded_keys.add(name)
            else:
                model_weights.append((name, tensor))
        loaded = self.model.load_weights(model_weights)
        loaded_keys.update(f"model.{k}" for k in loaded)
        return loaded_keys
