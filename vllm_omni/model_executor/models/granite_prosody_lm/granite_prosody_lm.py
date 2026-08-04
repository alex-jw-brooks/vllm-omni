# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM - Omni wrapper around GraniteMoeHybridForCausalLM.

First stage in the ProsodyLM pipeline: ingests text, produces prosody
latents for the downstream TTS stage. Runs as LLM_AR or LLM_GENERATION
depending on the pipeline config (AR vs NAR).
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
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .configuration_granite_prosody_lm import GraniteProsodyLMConfig

logger = init_logger(__name__)


class GraniteProsodyLMForConditionalGeneration(nn.Module):
    has_preprocess = True
    has_postprocess = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = cast(GraniteProsodyLMConfig, vllm_config.model_config.hf_config)
        self.vllm_config = vllm_config

        if self.config.simple_stack != 0:
            raise ValueError(f"Only simple_stack mode 0 is supported, got {self.config.simple_stack}")

        # FIXME: wrapping ForCausalLM instead of the inner Model to avoid
        # reimplementing lm_head/logits_processor/load_weights, but this
        # triggers a spurious "SupportsLoRA.lora_manager not implemented"
        # warning. Consider wrapping GraniteMoeHybridModel directly.
        self.language_model = GraniteMoeHybridForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model.forward(
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
        return self.language_model.compute_logits(hidden_states)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        # TODO: format T1/T2 prompt with normalized text + speaker token
        return input_ids, input_embeds, {}

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: parse prosody tokens → per-word arrays for StyleTTS2
        return {}

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
