# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM pipeline config.

  3 stages — text norm (NLE/CTC) → prosody (NAR) → StyleTTS2

Each LLM stage uses a separate pre-merged checkpoint.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.granite_prosody_lm"

_STAGE_TEXT_NORM = StagePipelineConfig(
    stage_id=0,
    model_stage="text_norm",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(),
    owns_tokenizer=True,
    engine_output_type="latent",
    model_subdir="stage0_text_norm_nle",
    sampling_constraints={"detokenize": False},
)

_STAGE_PROSODY = StagePipelineConfig(
    stage_id=1,
    model_stage="prosody",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(0,),
    engine_output_type="latent",
    model_subdir="stage1_prosody",
    custom_process_input_func=(f"{_PROC}.process_text_norm_to_prosody"),
    sampling_constraints={"detokenize": False},
)

_STAGE_TTS = StagePipelineConfig(
    stage_id=2,
    model_stage="styletts2",
    model_arch="GraniteStyleTTS2Decoder",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(1,),
    engine_output_type="audio",
    model_subdir="stage2_tts",
    custom_process_input_func=(f"{_PROC}.process_prosody_to_tts"),
    final_output=True,
    final_output_type="audio",
    sampling_constraints={"detokenize": False},
)

GRANITE_PROSODY_LM_PIPELINE = PipelineConfig(
    model_type="granite_prosody_lm",
    default_deploy_config_name="granite_prosody_lm_tts.yaml",
    model_arch="GraniteProsodyLMForConditionalGeneration",
    stages=(_STAGE_TEXT_NORM, _STAGE_PROSODY, _STAGE_TTS),
)
