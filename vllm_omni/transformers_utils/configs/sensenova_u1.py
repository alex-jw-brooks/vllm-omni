# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace-style configuration classes for SenseNova-U1."""

from transformers import AutoConfig, PretrainedConfig

_SENSENOVA_U1_LLM_HIDDEN_DEFAULT = 4096


class SenseNovaU1LLMConfig(PretrainedConfig):
    """Qwen3-based LLM backbone config with 3D RoPE and MoT extensions."""

    model_type = "sensenova_u1_llm"

    def __init__(
        self,
        hidden_size: int = _SENSENOVA_U1_LLM_HIDDEN_DEFAULT,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        vocab_size: int = 152064,
        rope_theta: float = 1000000.0,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings: int = 40960,
        max_position_embeddings_hw: int = 10000,
        attention_bias: bool = True,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings = max_position_embeddings
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self.attention_bias = attention_bias
        self.layer_types = layer_types if layer_types is not None else ["full_attention"] * num_hidden_layers
        kwargs["tie_word_embeddings"] = tie_word_embeddings
        super().__init__(**kwargs)


class SenseNovaU1VisionConfig(PretrainedConfig):
    """Vision embedding config (2D RoPE + conv patch embed, no transformer)."""

    model_type = "sensenova_u1_vision"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 1024,
        llm_hidden_size: list[int] | None = None,
        downsample_ratio: list[float] | None = None,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size if llm_hidden_size is not None else [_SENSENOVA_U1_LLM_HIDDEN_DEFAULT]
        self.downsample_ratio = downsample_ratio if downsample_ratio is not None else [0.5]
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        super().__init__(**kwargs)


class SenseNovaU1Config(PretrainedConfig):
    """Top-level composite config for SenseNova-U1.

    Nests ``llm_config`` and ``vision_config`` sub-configs alongside the
    flow-matching / diffusion parameters.  When constructed from a dict
    (e.g. via ``from_pretrained``), sub-dicts are automatically promoted
    to their typed config objects.
    """

    model_type = "sensenova_u1"

    def __init__(
        self,
        llm_config: dict | SenseNovaU1LLMConfig | None = None,
        vision_config: dict | SenseNovaU1VisionConfig | None = None,
        downsample_ratio: float = 0.5,
        template: str = "neo1_0",
        fm_head_layers: int = 2,
        fm_head_dim: int = 4096,
        fm_head_mlp_ratio: float = 1.0,
        use_pixel_head: bool = False,
        noise_scale: float = 1.0,
        noise_scale_mode: str = "none",
        noise_scale_base_image_seq_len: int = 256,
        noise_scale_max_value: float = 10.0,
        add_noise_scale_embedding: bool = False,
        time_schedule: str = "standard",
        time_shift_type: str = "exponential",
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        concat_time_token_num: int = 0,
        t_eps: float = 0.02,
        **kwargs,
    ):
        if isinstance(llm_config, dict):
            llm_config = SenseNovaU1LLMConfig(**llm_config)
        self.llm_config = llm_config or SenseNovaU1LLMConfig()

        if isinstance(vision_config, dict):
            vision_config = SenseNovaU1VisionConfig(**vision_config)
        self.vision_config = vision_config or SenseNovaU1VisionConfig()

        self.downsample_ratio = downsample_ratio
        self.template = template
        self.fm_head_layers = fm_head_layers
        self.fm_head_dim = fm_head_dim
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.use_pixel_head = use_pixel_head
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.noise_scale_max_value = noise_scale_max_value
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.time_schedule = time_schedule
        self.time_shift_type = time_shift_type
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.concat_time_token_num = concat_time_token_num
        self.t_eps = t_eps
        super().__init__(**kwargs)


AutoConfig.register("sensenova_u1", SenseNovaU1Config)

__all__ = [
    "SenseNovaU1Config",
    "SenseNovaU1LLMConfig",
    "SenseNovaU1VisionConfig",
]
