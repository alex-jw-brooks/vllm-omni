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

from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.granitemoehybrid import (
    GraniteMoeHybridForCausalLM,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.transformers_utils.configs.granite_prosody_lm import (
    GraniteProsodyLMConfig,
)

logger = init_logger(__name__)

_NUM_PROSODY_HEADS = 6


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
            if self.config.emotion_control != 0:
                raise ValueError(
                    "Emotion NAR heads are not implemented. emotion_control must be 0 for the prosody stage."
                )
            num_codebook = self.config.num_end_id - self.config.num_start_id
            if num_codebook <= 0:
                raise ValueError(
                    f"Invalid prosody codebook range: "
                    f"num_start_id={self.config.num_start_id}, "
                    f"num_end_id={self.config.num_end_id}."
                )
            self.model.nar_heads = NARHeads(
                self.config.hidden_size,
                num_codebook,
            )

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model.forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if self.model_stage == "prosody" and self.config.nar_mode:
            raise NotImplementedError(
                "compute_logits should not be called for NAR prosody stage. "
                "NAR decode uses model.nar_heads.forward_head() directly."
            )
        return self.model.compute_logits(hidden_states)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if self.model_stage == "text_norm":
            # T1 prompt formatting (chat template + [SEP_1]) is handled at
            # the API/tokenizer layer. Stop token ([SEP_NORM]) is set in the
            # deploy YAML. Pass through as-is.
            return input_ids, input_embeds, {}
        if self.model_stage == "prosody":
            # TODO: build masked annotation block for NAR decode
            raise NotImplementedError("Prosody preprocess not yet implemented.")
        raise ValueError(f"Unknown model_stage={self.model_stage!r}")

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.model_stage == "text_norm":
            # AR output tokens are handled by the framework. The stage input
            # processor (text_norm_to_prosody) transforms them for Stage 1.
            return {}
        if self.model_stage == "prosody":
            # TODO: decode annotation block → per-word prosody arrays
            raise NotImplementedError("Prosody postprocess not yet implemented.")
        raise ValueError(f"Unknown model_stage={self.model_stage!r}")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # Checkpoint layout (produced by export_for_vllm.py):
        #   model.*       — GraniteMoeHybrid transformer body
        #   lm_head.*     — LM head (text_norm uses this for AR decode)
        #   nar_heads.*   — 6 prosody prediction heads (prosody stage only)
        # NAR heads are attached to self.model so all three prefixes resolve
        # through ForCausalLM's AutoWeightsLoader without remapping.
        return self.model.load_weights(weights)
