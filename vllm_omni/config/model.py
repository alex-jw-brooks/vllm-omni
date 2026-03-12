from dataclasses import field
from typing import Any

from pydantic import ConfigDict, TypeAdapter
from transformers import PretrainedConfig
from vllm.config import ModelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config

import vllm_omni.model_executor.models as me_models

logger = init_logger(__name__)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig.

     This configuration class extends the base vLLM ModelConfig with
     omni-specific fields for multi-stage pipeline processing.

     Attributes:
         hf_config: The model's HF Transformers config (default: None)
         hf_text_config: The sub text_config of the model's hf_config (default: None)
         stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
         async_chunk: If set to True, perform async chunk
         model_stage: Stage type identifier, e.g., "thinker" or "talker"
             (default: "thinker")
         model_arch: Model architecture name
             (default: "Qwen2_5OmniForConditionalGeneration")
         worker_type: Model Type, e.g., "ar" or "generation"
         engine_output_type: Optional output type specification for the engine.
             Used to route outputs to appropriate processors (e.g., "image",
             "audio", "latents"). If None, output type is inferred.
         stage_connector_config: Stage connector configuration dictionary.
             Contains "name" (connector name), "extra" (extra connector config).
         task_type: Default task type for TTS models (CustomVoice, VoiceDesign, or Base).
             If not specified, will be inferred from model path.


    The correct way to initialize this class is via vLLM config, as most
    of the logic for handling values is in the ModelConfig's __post_init__.

       Example:
         >>> config = OmniModelConfig.from_vllm_model_config(
         ...     vllm_config,
         ...     stage_id=0,
         ...     model_stage="thinker",
         ...     model_arch="Qwen2_5OmniForConditionalGeneration"
         ... )
    """

    # Fields that set init=False in ModelConfig; we explicitly require
    # these because we leverage the ModelConfig's post_init to handle
    # (most) settings outside of the Omni config options.
    hf_config: PretrainedConfig | None = None
    """The Hugging Face config of the model."""
    hf_text_config: PretrainedConfig | None = None
    """The Hugging Face config of the text model (same as hf_config for text models)."""

    stage_id: int = 0
    async_chunk: bool = False
    model_stage: str = "thinker"
    model_arch: str | None = None
    worker_type: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "SharedMemoryConnector",
            "extra": {},
        }
    )
    omni_kv_config: dict | None = None
    codec_frame_rate_hz: float | None = None
    task_type: str | None = None

    @property
    def registry(self):
        return me_models.OmniModelRegistry

    @property
    def architectures(self) -> list[str]:
        if self.model_arch is not None:
            return [self.model_arch]
        return super().architectures

    @property
    def embedding_size(self):
        if self.hf_config_name is not None:
            stage_config = getattr(self.hf_config, self.hf_config_name, None)
            override = getattr(stage_config, "embedding_size", None)
            if override is not None:
                return override
        return super().embedding_size

    def draw_hf_text_config(self):
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        if self.hf_config_name is None:
            return get_hf_text_config(self.hf_config)
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(self.hf_config, self.hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{self.hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(self.hf_config)

    def _patch_qwen3_tts(self):
        """Patches the value of `position_id_per_seconds` in Qwen3's
        TTS's talker_config into the this class's codec_frame_rate_hz.
        """
        talker_cfg = getattr(self.hf_config, "talker_config", None)
        if isinstance(talker_cfg, dict):
            pos_per_sec = talker_cfg.get("position_id_per_seconds")
        else:
            pos_per_sec = getattr(talker_cfg, "position_id_per_seconds", None)
        if pos_per_sec is not None:
            try:
                fps = float(pos_per_sec)
            except Exception:
                fps = None
            if fps is not None and fps > 0:
                self.codec_frame_rate_hz = fps

    def _maybe_override_text_config(self):
        """Override hf_text_config with omni-specific logic for multi-stage
        models (e.g., thinker_config, talker_config).
        """
        new_hf_text_config = self.draw_hf_text_config()
        if new_hf_text_config is not self.hf_text_config:
            self.hf_text_config = new_hf_text_config
            # Recalculate dependent attributes
            self.attention_chunk_size = getattr(self.hf_text_config, "attention_chunk_size", None)
            # Recalculate max_model_len since it depends on hf_text_config
            self.max_model_len = self.get_and_verify_max_len(self.original_max_model_len)
            # Reset sliding_window if needed
            if self.disable_sliding_window and self.hf_text_config is not None:
                self.hf_text_config.sliding_window = None

    @classmethod
    def from_vllm_model_config(cls, model_config: ModelConfig, **omni_kwargs):
        """Create OmniModelConfig from an existing vLLM ModelConfig
        and additional Omni specific kwargs.
        """
        # We only validate the omni specific kwargs here since ModelConfig has
        # already validated its fields; Creating through .__new__ and directly
        # validating through TypeAdapters allows us to best effort check the config
        # for correctness without double calling expensive validation and post
        # initialization from the superclass.
        cls._validate_omni_fields(**omni_kwargs)

        omni_cfg = object.__new__(cls)
        omni_cfg.__dict__.update(model_config.__dict__)
        omni_cfg.__dict__.update(omni_kwargs)

        if (
            omni_cfg.codec_frame_rate_hz is None
            and omni_cfg.model_arch == "Qwen3TTSTalkerForConditionalGenerationARVLLM"
        ):
            omni_cfg._patch_qwen3_tts()

        omni_cfg._maybe_override_text_config()

        if omni_cfg.hf_config is not None:
            omni_cfg.hf_config.architectures = omni_cfg.architectures

        return omni_cfg

    @classmethod
    def _validate_omni_fields(cls, **omni_kwargs):
        """Validate omni-specific fields; we use TypeAdapters here to quickly
        validate only omni kwargs to avoid rerunning validation on the
        ModelConfig.
        """
        omni_fields = set(cls.__dataclass_fields__) - set(ModelConfig.__dataclass_fields__)

        for key, value in omni_kwargs.items():
            if key not in omni_fields:
                raise ValueError(f"Unexpected omni kwarg: {key}")

            field_type = cls.__dataclass_fields__[key].type
            if field_type is not Any:
                TypeAdapter(field_type).validate_python(value)
