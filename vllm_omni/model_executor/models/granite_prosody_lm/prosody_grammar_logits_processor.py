# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM v1 logits processor adapter for ProsodyGrammarProcessor.

Registered via deploy YAML FQCN; creates a ProsodyGrammarProcessor
per-request from SamplingParams.extra_args["prosody_grammar"].

When extra_args is absent, creates a dynamic-mode processor that
enforces prosody structure without knowing num_words in advance.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .prosody_grammar import ProsodyGrammarProcessor

logger = init_logger(__name__)


class ProsodyGrammarLogitsProcessor(AdapterLogitsProcessor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        config = vllm_config.model_config.hf_config
        self._grammar_kwargs = {
            "num_start_id": config.num_start_id,
            "num_end_id": config.num_end_id,
            "sil_token_id": config.sil_token_id,
            "sep2_token_id": config.sep2_token_id,
            "emotion_control": config.emotion_control,
        }

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: Any,
    ) -> ProsodyGrammarProcessor | None:
        extra_args = params.extra_args if params else None
        if extra_args is not None and not isinstance(extra_args, dict):
            raise TypeError(f"Expected extra_args to be dict or None, got {type(extra_args)}")
        grammar_cfg = extra_args.get("prosody_grammar") if extra_args else None
        if grammar_cfg is not None:
            logger.info(
                "Activating ProsodyGrammarProcessor: num_words=%d",
                grammar_cfg["num_words"],
            )
            return ProsodyGrammarProcessor(
                num_words=grammar_cfg["num_words"],
                **self._grammar_kwargs,
            )
        logger.info("Activating ProsodyGrammarProcessor in dynamic mode")
        return ProsodyGrammarProcessor(
            num_words=None,
            **self._grammar_kwargs,
        )
