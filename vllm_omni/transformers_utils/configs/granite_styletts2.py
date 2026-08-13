# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Granite StyleTTS2 decoder (Stage 2 of ProsodyLM).

Registers GraniteStyleTTS2Config (model_type="granite_styletts2") so
``AutoConfig.from_pretrained("path/to/styletts2")`` returns the correct
config class. Fields mirror config_ft.yml model_params.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class GraniteStyleTTS2Config(PretrainedConfig):
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
        # Engine core requires LLM-style fields for ModelArchitectureConfig.
        # These are not used by StyleTTS2 but must be present and non-zero.
        self.hidden_size = hidden_dim
        self.num_attention_heads = 1
        self.num_hidden_layers = 1
        super().__init__(**kwargs)


AutoConfig.register("granite_styletts2", GraniteStyleTTS2Config)

__all__ = ["GraniteStyleTTS2Config"]
