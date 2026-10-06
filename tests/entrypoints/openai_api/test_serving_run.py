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
from vllm_omni.inputs.data import OmniPromptType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _next_stage_input(requests: list[EngineCoreRequest] | list[OmniPromptType]) -> NextStageInputMessage:
    return NextStageInputMessage(
        request_id="req-run",
        source_stage_id=0,
        receiver_stage_id=1,
        requests=requests,
        submit_kwargs=None,
        stage_output=None,
        sampling_params_list=[],
        final_stage_id=1,
        final_output_stage_ids=[1],
    )


def test_stage_input_round_trips_llm_receiver():
    """Ensure an LLM receiver's requests, including omni-only fields, survive encoding."""
    stage_types = ("llm", "llm")
    request = build_engine_core_request_from_tokens("req-run", {"prompt_token_ids": [7]}, SamplingParams(max_tokens=9))
    request.additional_information = serialize_additional_information({"codes": torch.arange(2)})
    stage_input = _next_stage_input([request])
    assert decode_stage_input(encode_stage_input(stage_input), stage_types) == stage_input


def test_stage_input_round_trips_diffusion_receiver():
    """Ensure tensors inside a diffusion prompt survive encoding."""
    hidden_states = torch.ones(2, 4)
    stage_types = ("llm", "diffusion")
    stage_input = _next_stage_input([{"extra": {"hidden_states": hidden_states}}])
    decoded = decode_stage_input(encode_stage_input(stage_input), stage_types)
    torch.testing.assert_close(decoded.requests[0]["extra"]["hidden_states"], hidden_states)
