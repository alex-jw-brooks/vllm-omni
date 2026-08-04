# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM config registration with transformers AutoConfig.

Registers GraniteProsodyLMConfig (model_type="granite_prosody_lm") so that
``AutoConfig.from_pretrained(...)`` returns the correct config class.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.granite_prosody_lm.configuration_granite_prosody_lm import (
    GraniteProsodyLMConfig,
)

AutoConfig.register("granite_prosody_lm", GraniteProsodyLMConfig)

__all__ = [
    "GraniteProsodyLMConfig",
]
