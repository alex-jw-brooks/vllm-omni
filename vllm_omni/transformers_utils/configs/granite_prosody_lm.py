# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Granite ProsodyLM (TTS via prosody prediction).

Extends GraniteMoeHybridConfig with prosody-specific fields. The backbone
is a standard GraniteMoeHybrid; this config adds the inference knobs for
AR text normalization (Stage 0) and NAR prosody prediction (Stage 1).
"""

from __future__ import annotations

from transformers import AutoConfig, GraniteMoeHybridConfig, PretrainedConfig


class GraniteProsodyLMConfig(GraniteMoeHybridConfig):
    model_type = "granite_prosody_lm"

    def __init__(
        self,
        compact_preamble: bool = True,
        # Token IDs
        mask_token_id: int = 0,
        num_start_id: int = 0,
        num_end_id: int = 0,
        sil_token_id: int = 0,
        sep1_token_id: int = 0,
        sep2_token_id: int = 0,
        sep_norm_token_id: int = 0,
        sep_f0_token_id: int = 0,
        # NAR decode
        nar_tiers: list | None = None,
        nar_iterations: int = 2,
        nar_iterations_per_tier: list | None = None,
        nar_global_dims: list | None = None,
        nar_ensure_self_attend: bool = True,
        nar_history_causal: bool = True,
        nar_max_history: int = 1,
        nar_temperature: float = 0.2,
        # Speaker preamble (pre-tokenized by export script)
        speaker_prefix_ids: list | None = None,
        speaker_suffix_ids: list | None = None,
        default_f0_bin: int = 0,
        # CTC text normalization head
        ctc_head_vocab_size: int = 0,
        ctc_blank_index: int = 0,
        ctc_editor_copy_op: bool = False,
        ctc_copy_index: int | None = None,
        blank_token_id: int = 0,
        ctc_slots_per_gap: int = 3,
        ctc_steps: int = 1,
        # Emotion control (always 0 — passed through to preamble layout)
        emotion_control: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.compact_preamble = compact_preamble

        self.mask_token_id = mask_token_id
        self.num_start_id = num_start_id
        self.num_end_id = num_end_id
        self.sil_token_id = sil_token_id
        self.sep1_token_id = sep1_token_id
        self.sep2_token_id = sep2_token_id
        self.sep_norm_token_id = sep_norm_token_id
        self.sep_f0_token_id = sep_f0_token_id

        self.nar_tiers = nar_tiers or []
        self.nar_iterations = nar_iterations
        self.nar_iterations_per_tier = nar_iterations_per_tier or []
        self.nar_global_dims = nar_global_dims or []
        self.nar_ensure_self_attend = nar_ensure_self_attend
        self.nar_history_causal = nar_history_causal
        self.nar_max_history = nar_max_history
        self.nar_temperature = nar_temperature

        self.speaker_prefix_ids = speaker_prefix_ids or []
        self.speaker_suffix_ids = speaker_suffix_ids or []
        self.default_f0_bin = default_f0_bin

        self.emotion_control = emotion_control

        self.ctc_head_vocab_size = ctc_head_vocab_size
        self.ctc_blank_index = ctc_blank_index
        self.ctc_editor_copy_op = ctc_editor_copy_op
        self.ctc_copy_index = ctc_copy_index
        self.blank_token_id = blank_token_id
        self.ctc_slots_per_gap = ctc_slots_per_gap
        self.ctc_steps = ctc_steps


class GraniteStyleTTS2Config(PretrainedConfig):
    """Config for Granite StyleTTS2 decoder (Stage 2 of ProsodyLM pipeline).

    Fields mirror config_ft.yml model_params.
    """

    model_type = "granite_styletts2"

    def __init__(
        self,
        hidden_dim: int = 512,
        style_dim: int = 128,
        dim_in: int = 64,
        n_layer: int = 3,
        n_token: int = 178,
        max_dur: int = 50,
        n_mels: int = 80,
        dropout: float = 0.2,
        sr: int = 24000,
        resblock_kernel_sizes: list[int] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_initial_channel: int = 512,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.dim_in = dim_in
        self.n_layer = n_layer
        self.n_token = n_token
        self.max_dur = max_dur
        self.n_mels = n_mels
        self.dropout = dropout
        self.sr = sr
        self.resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        self.upsample_rates = upsample_rates or [10, 5, 3, 2]
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes or [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ]
        self.upsample_kernel_sizes = upsample_kernel_sizes or [20, 10, 6, 4]
        self.hidden_size = hidden_dim
        self.num_attention_heads = 1
        self.num_hidden_layers = 1
        super().__init__(**kwargs)


AutoConfig.register("granite_prosody_lm", GraniteProsodyLMConfig)
AutoConfig.register("granite_styletts2", GraniteStyleTTS2Config)

__all__ = ["GraniteProsodyLMConfig", "GraniteStyleTTS2Config"]
