# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline registry and factory for vllm-omni.

``OMNI_PIPELINES`` maps each ``model_type`` to either a ``PipelineConfig``
instance or a resolver callable that accepts an optional HF config and returns
a ``PipelineConfig``.

To add a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. If the model needs to support several configurations, e.g., because some
       stages are optional, implement a resolver that consumes the HF config
       and returns a ``PipelineConfig``.
    3. Update the registry to map the key to the new config object (in the case
       of new keys) or to the resolver func.

NOTE: Single-stage diffusion models continue to use the
``_create_default_diffusion_stage_cfg`` fallback in
``async_omni_engine.py``; for now we do not add them to registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

from transformers import PreTrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageType,
    _warn_deprecated_kwargs,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.config.yaml_util import create_config
from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_PIPELINE
from vllm_omni.model_executor.models.bagel.pipeline import (
    BAGEL_PIPELINE,
    BAGEL_SINGLE_STAGE_PIPELINE,
    BAGEL_THINK_PIPELINE,
)
from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_PIPELINE
from vllm_omni.model_executor.models.covo_audio.pipeline import COVO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.dreamzero.pipeline import DREAMZERO_PIPELINE
from vllm_omni.model_executor.models.dynin_omni.pipeline import DYNIN_OMNI_PIPELINE
from vllm_omni.model_executor.models.fish_speech.pipeline import FISH_SPEECH_PIPELINE
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE
from vllm_omni.model_executor.models.glm_tts.pipeline import GLM_TTS_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import HIGGS_AUDIO_V2_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v3.pipeline import HIGGS_AUDIO_V3_PIPELINE
from vllm_omni.model_executor.models.hunyuan_image3.pipeline import (
    HUNYUAN_IMAGE3_AR_PIPELINE,
    HUNYUAN_IMAGE3_DIT_PIPELINE,
    HUNYUAN_IMAGE3_PIPELINE,
)
from vllm_omni.model_executor.models.indextts2.pipeline import INDEXTTS2_PIPELINE
from vllm_omni.model_executor.models.lance.pipeline import LANCE_PIPELINE
from vllm_omni.model_executor.models.mimo_audio.pipeline import MIMO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.ming_flash_omni.pipeline import (
    MING_FLASH_OMNI_IMAGE_PIPELINE,
    MING_FLASH_OMNI_PIPELINE,
    MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    MING_FLASH_OMNI_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.ming_tts.pipeline import MING_TTS_PIPELINE
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.models.moss_tts.pipeline import MOSS_TTS_PIPELINE, MOSS_TTS_REALTIME_PIPELINE
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import MOSS_TTS_NANO_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    QWEN2_5_OMNI_PIPELINE,
    QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import resolve_qwen3_omni_pipeline
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.voxcpm2.pipeline import VOXCPM2_PIPELINE
from vllm_omni.model_executor.models.voxtral_tts.pipeline import VOXTRAL_TTS_PIPELINE

logger = init_logger(__name__)

PipelineResolverFunc: TypeAlias = Callable[[PreTrainedConfig | None], PipelineConfig | None]

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc] = {
    "aura_omni": AURA_OMNI_PIPELINE,
    "qwen2_5_omni": QWEN2_5_OMNI_PIPELINE,
    "qwen2_5_omni_thinker_only": QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
    "qwen3_omni_moe": resolve_qwen3_omni_pipeline,
    "qwen3_tts": QWEN3_TTS_PIPELINE,
    "covo_audio": COVO_AUDIO_PIPELINE,
    "bagel": BAGEL_PIPELINE,
    "bagel_think": BAGEL_THINK_PIPELINE,
    "bagel_single_stage": BAGEL_SINGLE_STAGE_PIPELINE,
    "lance": LANCE_PIPELINE,
    "dreamzero": DREAMZERO_PIPELINE,
    "glm_image": GLM_IMAGE_PIPELINE,
    "hunyuan_image_3_moe": HUNYUAN_IMAGE3_PIPELINE,
    "hunyuan_image3_ar": HUNYUAN_IMAGE3_AR_PIPELINE,
    "hunyuan_image3_dit": HUNYUAN_IMAGE3_DIT_PIPELINE,
    "voxcpm2": VOXCPM2_PIPELINE,
    "cosyvoice3": COSYVOICE3_PIPELINE,
    "mimo_audio": MIMO_AUDIO_PIPELINE,
    "ming_tts": MING_TTS_PIPELINE,
    "voxtral_tts": VOXTRAL_TTS_PIPELINE,
    "glm_tts": GLM_TTS_PIPELINE,
    "fish_qwen3_omni": FISH_SPEECH_PIPELINE,
    "ming_flash_omni": MING_FLASH_OMNI_PIPELINE,
    "ming_flash_omni_tts": MING_FLASH_OMNI_TTS_PIPELINE,
    "ming_flash_omni_thinker_only": MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    "ming_flash_omni_image": MING_FLASH_OMNI_IMAGE_PIPELINE,
    "moss_tts_nano": MOSS_TTS_NANO_PIPELINE,
    "moss_tts_delay": MOSS_TTS_PIPELINE,
    "moss_tts_realtime": MOSS_TTS_REALTIME_PIPELINE,
    "minicpmo_4_5": MINICPMO_4_5_PIPELINE,
    "higgs_audio_v2": HIGGS_AUDIO_V2_PIPELINE,
    "higgs_multimodal_qwen3": HIGGS_AUDIO_V3_PIPELINE,
    "dynin_omni": DYNIN_OMNI_PIPELINE,
    "indextts2": INDEXTTS2_PIPELINE,
}


def resolve_pipeline_config(model_type: str, hf_config: PreTrainedConfig | None = None) -> PipelineConfig | None:
    if model_type not in OMNI_PIPELINES:
        return None
    obj = OMNI_PIPELINES[model_type]
    return obj(hf_config) if callable(obj) else obj


class StageConfigFactory:
    """Factory that loads pipeline YAML and merges CLI overrides.

    Handles both single-stage and multi-stage models.

    Pipelines are declared in ``vllm_omni/config/pipeline_registry.py`` and
    where keys in OMNI_PIPELINES map to either a PipelineConfig, or a callable
    which accepts a Transformers config as an arg & resolves to a PipelineConfig.

    NOTE: Models with generic HF ``model_type`` collisions (e.g. MiMo Audio
    reports ``qwen2``) should declare ``hf_architectures=(...)`` on their
    ``PipelineConfig`` so the factory can disambiguate via ``hf_config.architectures``.
    """

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig] | None:
        """Load pipeline + deploy config, merge with CLI overrides.

        Checks OMNI_PIPELINES first, since supported models should be explicitly
        registered. If a model is not registered in OMNI_PIPELINES, tries to fall
        back to using the Transformers config & finding pipelines that have overlapping
        supported architectures.
        """
        _warn_deprecated_kwargs(deprecated_kwargs)

        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        if trust_remote_code is None:
            trust_remote_code = False

        # --- New path: check pipeline registry by model_type first ---
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type and model_type in OMNI_PIPELINES:
            pipeline_cfg = resolve_pipeline_config(model_type, hf_config)
            if pipeline_cfg is not None:
                return cls._create_from_registry(
                    model_type,
                    pipeline_cfg,
                    cli_overrides,
                    deploy_config_path,
                )

        # --- HF architecture fallback: some models report a generic
        # model_type that collides with another model. Match by the
        # hf_architectures declared on each registered PipelineConfig.
        if hf_config is not None:
            logger.warning("Inferred model type %s is not registered to an Omni pipeline", model_type)
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered in OMNI_PIPELINES.values():
                    if isinstance(registered, PipelineConfig) and hf_archs.intersection(registered.hf_architectures):
                        return cls._create_from_registry(
                            registered.model_type,
                            registered,
                            cli_overrides,
                            deploy_config_path,
                        )

        raise ValueError(
            f"Unable to create model; Model type {model_type} is not registered in OMNI_PIPELINES,"
            f" and hf_config of type {type(hf_config)}"
        )

    @classmethod
    def _create_from_registry(
        cls,
        model_type: str,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig]:
        """Create StageConfigs from pipeline registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.
        """
        _warn_deprecated_kwargs(deprecated_kwargs)

        # Resolve deploy config path
        if deploy_config_path is None:
            deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
        else:
            deploy_path = Path(deploy_config_path)

        if not deploy_path.exists():
            logger.warning(
                "Deploy config not found: %s — using pipeline defaults only",
                deploy_path,
            )
            deploy_cfg = DeployConfig()
        else:
            deploy_cfg = load_deploy_config(deploy_path)

        cli_async_chunk = cli_overrides.get("async_chunk")
        if cli_async_chunk is not None:
            deploy_cfg.async_chunk = bool(cli_async_chunk)

        stages = merge_pipeline_deploy(pipeline_cfg, deploy_cfg, cli_overrides)

        explicit_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        for stage in stages:
            stage.runtime_overrides = cls._merge_cli_overrides(stage, explicit_overrides)

        return stages

    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Single-stage diffusion - no YAML needed.

        Creates a default diffusion stage configuration for single-stage
        diffusion models. Returns a legacy OmegaConf-compatible dict for
        backward compatibility with OmniStage.

        Args:
            kwargs: Engine arguments from CLI/API.

        Returns:
            List containing a single config dict for the diffusion stage.
        """
        # Calculate devices based on parallel config
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"

        engine_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("parallel_config",):
                continue
            engine_args[key] = value

        # Serialize parallel_config as dict for OmegaConf. Test helpers
        # sometimes pass SimpleNamespace rather than a dataclass instance.
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            if dataclasses.is_dataclass(parallel_config) and not isinstance(parallel_config, type):
                engine_args["parallel_config"] = asdict(parallel_config)
            elif hasattr(parallel_config, "__dict__"):
                engine_args["parallel_config"] = dict(vars(parallel_config))
            else:
                engine_args["parallel_config"] = parallel_config

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        engine_args.setdefault("max_num_seqs", 1)

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": StageType.DIFFUSION.value,
            "runtime": {
                "process": True,
                "devices": devices,
            },
            "engine_args": create_config(engine_args),
            "final_output": True,
            "final_output_type": "image",
        }

        return [config_dict]

    # Keys consumed as explicit StageConfig fields — everything else is
    # passed through via yaml_extras.
    _KNOWN_STAGE_KEYS: set[str] = {
        "stage_id",
        "model_stage",
        "stage_type",
        "input_sources",
        "engine_input_source",
        "custom_process_input_func",
        "final_output",
        "final_output_type",
        "worker_type",
        "scheduler_cls",
        "hf_config_name",
        "is_comprehension",
        "engine_args",
        "runtime",
    }

    @classmethod
    def _auto_detect_model_type(cls, model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            Tuple of (model_type, hf_config). Both may be None on failure.
        """
        hf_config = None

        try:
            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            return hf_config.model_type, hf_config
        except Exception as e:
            logger.debug(f"`get_config` failed for {e}; Falling back to raw config.json path")

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            config_dict = get_hf_file_to_dict("config.json", model, revision=None)
            if config_dict:
                if "model_type" in config_dict:
                    return config_dict["model_type"], None
                # VoxCPM2-style configs use singular ``architecture`` rather
                # than HF's standard ``model_type`` / ``architectures``. Accept
                # it as a fallback so the pipeline registry can still match.
                if "architecture" in config_dict and isinstance(config_dict["architecture"], str):
                    return config_dict["architecture"], None
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")

        # Fallback for diffusers-style models: check model_index.json.
        # Some models (e.g. GLM-Image) have no root config.json but ship a
        # model_index.json with _class_name that maps to a pipeline key via
        # PipelineConfig.diffusers_class_name.
        try:
            model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
            if model_index and "_class_name" in model_index:
                class_name = model_index["_class_name"]
                for obj in OMNI_PIPELINES.values():
                    # If we have a resolver, call it with the optional hf_config
                    # to get the default pipeline config for this key
                    pipeline_cfg = obj(hf_config) if callable(obj) else obj
                    if pipeline_cfg.diffusers_class_name == class_name:
                        logger.info(
                            "Detected pipeline %r from model_index.json (_class_name=%r)",
                            pipeline_cfg.model_type,
                            class_name,
                        )
                        return pipeline_cfg.model_type, None
        except Exception as e:
            logger.debug(f"Failed to detect model type for diffusers-style models: {e}")

        # Final fallback: some models (e.g. CosyVoice3) ship an empty
        # config.json and rely on naming conventions. Match the model path
        # basename against registered pipeline keys — longest match wins
        # so "cosyvoice3" (length 10) beats "cosyvoice" (length 9).
        model_lower = model.lower().replace("-", "").replace("_", "")
        best: str | None = None
        best_len = 0
        for registered_key in OMNI_PIPELINES.keys():
            candidate = registered_key.lower().replace("-", "").replace("_", "")
            if candidate and candidate in model_lower and len(candidate) > best_len:
                best = registered_key
                best_len = len(candidate)
        if best is not None:
            return best, None

        return None, None

    @classmethod
    def _merge_cli_overrides(
        cls,
        stage: StageConfig,
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge global and per-stage (``stage_N_*``) CLI overrides.

        Orchestrator-owned keys are filtered by ``build_stage_runtime_overrides``
        using ``OrchestratorArgs`` as the single source of truth; unknown
        server/uvicorn keys are dropped downstream by
        ``filter_dataclass_kwargs(OmniEngineArgs, ...)``.
        """
        return build_stage_runtime_overrides(stage.stage_id, cli_overrides)
