# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Granite ProsodyLM (TTS via prosody prediction).

Extends GraniteMoeHybridConfig with prosody-specific fields. The backbone
is a standard GraniteMoeHybrid (dense, 0 MoE experts); this config adds
the TTS pipeline knobs: AR vs NAR mode, template mode, stacking mode, etc.
"""

from __future__ import annotations

from transformers import GraniteMoeHybridConfig


class GraniteProsodyLMConfig(GraniteMoeHybridConfig):
    model_type = "granite_prosody_lm"

    def __init__(
        self,
        nar_mode: bool = False,
        simple_stack: int = 0,
        template_mode: str = "T1",
        use_fractional_posids: bool = False,
        emotion_control: int = 0,
        num_prosody_dims: int = 6,
        # Grammar processor token IDs (from the merged tokenizer)
        num_start_id: int = 0,
        num_end_id: int = 0,
        sil_token_id: int = 0,
        sep2_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nar_mode = nar_mode
        self.simple_stack = simple_stack
        self.template_mode = template_mode
        self.use_fractional_posids = use_fractional_posids
        self.emotion_control = emotion_control
        self.num_prosody_dims = num_prosody_dims
        self.num_start_id = num_start_id
        self.num_end_id = num_end_id
        self.sil_token_id = sil_token_id
        self.sep2_token_id = sep2_token_id
