# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pybase64 as base64
import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder
from vllm_omni.engine.messages import NextStageInputMessage, OutputMessage
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai.protocol.run import RunRequest
from vllm_omni.entrypoints.openai.serving_run import ServingRun, decode_stage_input, encode_payload
from vllm_omni.errors import OmniClientError
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _next_stage_input(requests: list[EngineCoreRequest] | None = None) -> NextStageInputMessage:
    return NextStageInputMessage(
        request_id="req-run",
        source_stage_id=0,
        receiver_stage_id=1,
        requests=requests or [],
        submit_kwargs=None,
        stage_output=None,
        sampling_params_list=[],
        final_stage_id=1,
        final_output_stage_ids=[1],
    )


def _engine_client(response: NextStageInputMessage | OutputMessage) -> SimpleNamespace:
    """A two-stage LLM engine client whose run calls return ``response``."""
    return SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="llm")] * 2,
        run_entry_stage=AsyncMock(return_value=response),
        run_downstream_stage=AsyncMock(return_value=response),
    )


def test_stage_input_round_trips_llm_receiver():
    """Ensure an LLM receiver's requests, including omni-only fields, survive encoding."""
    request = build_engine_core_request_from_tokens("req-run", {"prompt_token_ids": [7]}, SamplingParams(max_tokens=9))
    request.additional_information = serialize_additional_information({"codes": torch.arange(2)})
    stage_input = _next_stage_input([request])
    assert decode_stage_input(encode_payload(stage_input), ("llm", "llm")) == stage_input


@pytest.mark.asyncio
async def test_call_runs_stage_with_final_only_output_kind():
    """Ensure a call runs its stage with FINAL_ONLY params so the final output is complete."""
    engine_client = _engine_client(_next_stage_input())
    await ServingRun(engine_client).run(RunRequest(stage_input={"prompt": "hi"}), request_id="run-entry")
    sampling_params_list = engine_client.run_entry_stage.call_args.args[1]
    assert all(params.output_kind == RequestOutputKind.FINAL_ONLY for params in sampling_params_list)


@pytest.mark.asyncio
async def test_non_final_call_returns_next_stage_id_and_stage_input():
    """Ensure a call that yields returns the next stage's id and its encoded input."""
    next_stage_input = _next_stage_input()
    serving = ServingRun(_engine_client(next_stage_input))
    response = await serving.run(RunRequest(stage_input={"prompt": "hi"}), request_id="run-entry")
    assert response.stage_id == next_stage_input.receiver_stage_id
    assert decode_stage_input(response.stage_input, serving.stage_types) == next_stage_input


@pytest.mark.asyncio
async def test_final_call_returns_encoded_output():
    """Ensure the final stage's call returns its encoded output."""
    final_output = OmniRequestOutput(request_id="run-final", stage_id=1)
    serving = ServingRun(
        _engine_client(OutputMessage(request_id="run-final", stage_id=1, engine_outputs=final_output, finished=True))
    )
    request = RunRequest(stage_id=1, stage_input=encode_payload(_next_stage_input()))
    response = await serving.run(request, request_id="run-final")
    assert OmniMsgpackDecoder().decode(base64.b64decode(response.output)) == final_output


@pytest.mark.asyncio
async def test_entry_stage_call_rejects_encoded_stage_input():
    """Ensure stage 0 rejects an encoded stage input instead of running it as a text prompt."""
    engine_client = _engine_client(_next_stage_input())
    request = RunRequest(stage_id=0, stage_input=encode_payload(_next_stage_input()))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_entry_stage.assert_not_awaited()


@pytest.mark.asyncio
async def test_downstream_stage_call_rejects_stage_input_built_for_another_stage():
    """Ensure a call is rejected when its stage_input was built for a different stage than its stage_id."""
    engine_client = _engine_client(_next_stage_input())
    stage_input = _next_stage_input()
    request = RunRequest(stage_id=stage_input.receiver_stage_id + 1, stage_input=encode_payload(stage_input))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_downstream_stage.assert_not_awaited()
