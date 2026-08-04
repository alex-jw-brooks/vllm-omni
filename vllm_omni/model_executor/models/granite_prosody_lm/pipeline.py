# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite ProsodyLM pipeline config.

Two Omni stages:
  0. granite_lm  - Granite LLM (text norm + prosody, same weights, called twice)
  1. styletts2   - waveform synthesis (LLM_GENERATION)
"""

from vllm.logger import init_logger

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    pipeline_cfg_resolver,
)
from vllm_omni.model_executor.models.granite_prosody_lm.configuration_granite_prosody_lm import (
    GraniteProsodyLMConfig,
)

logger = init_logger(__name__)

_PROC = "vllm_omni.model_executor.stage_input_processors.granite_prosody_lm"


def build_granite_prosody_pipeline(is_nar: bool) -> PipelineConfig:
    """Build the Granite Prosody pipeline config.

    The only difference between AR and NAR is the execution type of Stage 0:
    LLM_AR vs LLM_GENERATION.
    """
    lm_execution_type = StageExecutionType.LLM_GENERATION if is_nar else StageExecutionType.LLM_AR
    return PipelineConfig(
        model_type="granite_prosody_lm",
        default_deploy_config_name="granite_prosody_lm.yaml",
        model_arch="GraniteProsodyLMForConditionalGeneration",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="granite_lm",
                execution_type=lm_execution_type,
                input_sources=(),
                owns_tokenizer=True,
                engine_output_type="latent",
                # TODO: uncomment when StyleTTS2 stage is implemented
                # custom_process_next_stage_input_func=(
                #     f"{_PROC}.prosody_to_styletts2_full_payload"
                # ),
                final_output=True,
                sampling_constraints={
                    "detokenize": False,
                },
            ),
            # TODO: uncomment when StyleTTS2 stage is implemented
            # StagePipelineConfig(
            #     stage_id=1,
            #     model_stage="styletts2",
            #     model_arch="GraniteStyleTTS2Decoder",
            #     execution_type=StageExecutionType.LLM_GENERATION,
            #     input_sources=(0,),
            #     final_output=True,
            #     final_output_type="audio",
            #     engine_output_type="audio",
            #     sampling_constraints={"detokenize": True},
            # ),
        ),
    )


@pipeline_cfg_resolver(config_type=GraniteProsodyLMConfig)
def resolve_granite_prosody_pipeline(
    hf_config: GraniteProsodyLMConfig,
) -> PipelineConfig:
    """Select AR vs NAR pipeline based on the HF config's nar_mode field."""
    logger.info(f"Granite Prosody Pipeline: using NAR? {hf_config.nar_mode}")
    return build_granite_prosody_pipeline(hf_config.nar_mode)
