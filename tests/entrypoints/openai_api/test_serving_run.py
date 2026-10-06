# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine.messages import NextStageInputMessage
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai.serving_run import decode_stage_input, encode_stage_input
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _next_stage_input(
    requests: list[EngineCoreRequest] | list[OmniPromptType], sampling_params_list: list[OmniSamplingParams]
) -> NextStageInputMessage:
    return NextStageInputMessage(
        request_id="req-run",
        source_stage_id=0,
        receiver_stage_id=1,
        requests=requests,
        submit_kwargs=None,
        stage_output=None,
        sampling_params_list=sampling_params_list,
        final_stage_id=1,
        final_output_stage_ids=[1],
    )


def test_stage_input_round_trips_llm_receiver():
    """Ensure an LLM receiver's requests, including omni-only fields, and its stage params survive encoding."""
    stage_types = ("llm", "llm")
    sampling_params_list = [SamplingParams(), SamplingParams(max_tokens=9)]
    request = build_engine_core_request_from_tokens("req-run", {"prompt_token_ids": [7]}, sampling_params_list[-1])
    request.additional_information = serialize_additional_information({"codes": torch.arange(2)})
    stage_input = _next_stage_input([request], sampling_params_list)
    assert decode_stage_input(encode_stage_input(stage_input), stage_types) == stage_input


def test_stage_input_round_trips_diffusion_receiver():
    """Ensure tensors inside a diffusion prompt and LoRA in diffusion params survive encoding."""
    hidden_states = torch.ones(2, 4)
    stage_types = ("llm", "diffusion")
    sampling_params_list = [
        SamplingParams(),
        OmniDiffusionSamplingParams(lora_request=LoRARequest(lora_name="adapter", lora_int_id=1, lora_path="/adapter")),
    ]
    stage_input = _next_stage_input([{"extra": {"hidden_states": hidden_states}}], sampling_params_list)
    decoded = decode_stage_input(encode_stage_input(stage_input), stage_types)
    torch.testing.assert_close(decoded.requests[0]["extra"]["hidden_states"], hidden_states)
    assert decoded.sampling_params_list == stage_input.sampling_params_list
