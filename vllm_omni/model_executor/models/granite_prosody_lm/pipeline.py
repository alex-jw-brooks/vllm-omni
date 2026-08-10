# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM pipeline configs.

Three pipeline variants built from a shared builder:
  1 stage  — text normalization only (AR)
  2 stages — text norm (AR) → prosody (AR or NAR)
  3 stages — text norm → prosody → StyleTTS2 (deferred)

Each LLM stage uses a separate pre-merged checkpoint.
"""

import dataclasses

from vllm.logger import init_logger

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    pipeline_cfg_resolver,
)
from vllm_omni.transformers_utils.configs.granite_prosody_lm import (
    GraniteProsodyLMConfig,
)

logger = init_logger(__name__)

_PROC = "vllm_omni.model_executor.stage_input_processors.granite_prosody_lm"

_FINAL_OUTPUT_TYPE = {
    "text_norm": "text",
    "prosody": "latent",
    "styletts2": "audio",
}

# --- Base stage definitions ---

_STAGE_TEXT_NORM = StagePipelineConfig(
    stage_id=0,
    model_stage="text_norm",
    execution_type=StageExecutionType.LLM_AR,
    input_sources=(),
    owns_tokenizer=True,
    engine_output_type="latent",
    sampling_constraints={"detokenize": False},
)

_STAGE_PROSODY = StagePipelineConfig(
    stage_id=1,
    model_stage="prosody",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(0,),
    engine_output_type="latent",
    custom_process_next_stage_input_func=(f"{_PROC}.text_norm_to_prosody"),
    sampling_constraints={"detokenize": False},
)

_STAGE_TTS = StagePipelineConfig(
    stage_id=2,
    model_stage="styletts2",
    model_arch="GraniteStyleTTS2Decoder",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(1,),
    engine_output_type="audio",
    sampling_constraints={"detokenize": True},
)


# --- Per-stage builders ---


def _text_norm_stage() -> StagePipelineConfig:
    return _STAGE_TEXT_NORM


def _prosody_stage(is_nar: bool) -> StagePipelineConfig:
    if not is_nar:
        return dataclasses.replace(
            _STAGE_PROSODY,
            execution_type=StageExecutionType.LLM_AR,
        )
    return _STAGE_PROSODY


def _tts_stage() -> StagePipelineConfig:
    return _STAGE_TTS


# --- Pipeline builder ---


def build_granite_prosody_pipeline(
    n_stages: int,
    is_nar: bool,
) -> PipelineConfig:
    """Build a Granite Prosody pipeline with 1-3 stages.

    Args:
        n_stages: 1 = text_norm only, 2 = + prosody, 3 = + TTS.
        is_nar: If True, prosody stage uses NAR (LLM_GENERATION).
                If False, prosody stage uses AR (LLM_AR).
    """
    if n_stages > 2:
        raise ValueError(f"n_stages={n_stages} requires StyleTTS2 which is not yet ported.")

    stages = [_text_norm_stage()]
    if n_stages >= 2:
        stages.append(_prosody_stage(is_nar))
    if n_stages >= 3:
        stages.append(_tts_stage())

    last = stages[-1]
    out_type = _FINAL_OUTPUT_TYPE[last.model_stage]
    # detokenize override: only matters for text_norm as final stage
    # (flips False→True); redundant but harmless for prosody/TTS.
    stages[-1] = dataclasses.replace(
        last,
        final_output=True,
        final_output_type=out_type,
        engine_output_type=out_type,
        sampling_constraints={"detokenize": out_type != "latent"},
    )

    return PipelineConfig(
        model_type="granite_prosody_lm",
        default_deploy_config_name="granite_prosody_lm.yaml",
        model_arch="GraniteProsodyLMForConditionalGeneration",
        stages=tuple(stages),
    )


@pipeline_cfg_resolver(config_type=GraniteProsodyLMConfig)
def resolve_granite_prosody_pipeline(
    hf_config: GraniteProsodyLMConfig,
) -> PipelineConfig:
    # TODO: select n_stages from config / deploy settings.
    # For now, hardcode to 2 (text_norm + prosody).
    n_stages = 2
    logger.info(
        "Granite Prosody Pipeline: %d stages, NAR=%s",
        n_stages,
        hf_config.nar_mode,
    )
    return build_granite_prosody_pipeline(n_stages, hf_config.nar_mode)
