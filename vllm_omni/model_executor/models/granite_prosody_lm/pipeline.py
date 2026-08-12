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
    model_subdir="stage0_text_norm",
    custom_process_next_stage_input_func=(f"{_PROC}.text_norm_to_prosody"),
    sampling_constraints={"detokenize": False},
)

_STAGE_PROSODY = StagePipelineConfig(
    stage_id=1,
    model_stage="prosody",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(0,),
    engine_output_type="latent",
    model_subdir="stage1_prosody",
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


_STAGE_TEXT_NORM_NLE = StagePipelineConfig(
    stage_id=0,
    model_stage="text_norm",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(),
    owns_tokenizer=True,
    engine_output_type="latent",
    model_subdir="stage0_text_norm_nle",
    custom_process_next_stage_input_func=(f"{_PROC}.text_norm_to_prosody"),
    sampling_constraints={"detokenize": False},
)


def _text_norm_stage(is_nle: bool = False) -> StagePipelineConfig:
    return _STAGE_TEXT_NORM_NLE if is_nle else _STAGE_TEXT_NORM


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
    is_nle: bool = False,
) -> PipelineConfig:
    """Build a Granite Prosody pipeline with 1-3 stages.

    Args:
        n_stages: 1 = text_norm only, 2 = + prosody, 3 = + TTS.
        is_nar: If True, prosody stage uses NAR (LLM_GENERATION).
                If False, prosody stage uses AR (LLM_AR).
        is_nle: If True, text_norm uses NLE CTC (LLM_GENERATION).
                If False, text_norm uses standard AR.
    """
    if n_stages > 2:
        raise ValueError(f"n_stages={n_stages} requires StyleTTS2 which is not yet ported.")

    stages = [_text_norm_stage(is_nle)]
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

    if is_nle:
        deploy_yaml = "granite_prosody_lm_nle.yaml"
    elif is_nar:
        deploy_yaml = "granite_prosody_lm_nar.yaml"
    else:
        deploy_yaml = "granite_prosody_lm_ar.yaml"
    return PipelineConfig(
        model_type="granite_prosody_lm",
        default_deploy_config_name=deploy_yaml,
        model_arch="GraniteProsodyLMForConditionalGeneration",
        stages=tuple(stages),
    )


@pipeline_cfg_resolver(config_type=GraniteProsodyLMConfig)
def resolve_granite_prosody_pipeline(
    hf_config: GraniteProsodyLMConfig,
) -> PipelineConfig:
    n_stages = 2
    is_nle = getattr(hf_config, "ctc_text_norm", False)
    logger.info(
        "Granite Prosody Pipeline: %d stages, NAR=%s, NLE=%s",
        n_stages,
        hf_config.nar_mode,
        is_nle,
    )
    return build_granite_prosody_pipeline(n_stages, hf_config.nar_mode, is_nle)
