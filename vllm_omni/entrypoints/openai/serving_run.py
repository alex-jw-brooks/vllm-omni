# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import msgspec
import pybase64 as base64
from vllm.sampling_params import SamplingParams

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.messages import NextStageInputMessage
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams


def is_diffusion(stage_type: Literal["llm", "diffusion"]) -> bool:
    return stage_type == "diffusion"


def encode_stage_input(stage_input: NextStageInputMessage) -> str:
    """Encode a next stage input as base64 text for a JSON response."""
    return base64.b64encode(OmniMsgpackEncoder().encode(stage_input)).decode("ascii")


def decode_stage_input(data: str, stage_types: Sequence[Literal["llm", "diffusion"]]) -> NextStageInputMessage:
    """Decode a next stage input encoded by ``encode_stage_input``.

    The codec returns structs and dataclasses as plain containers, which we rebuild based on stage type.
    """
    fields = OmniMsgpackDecoder().decode(base64.b64decode(data))
    if not is_diffusion(stage_types[fields["receiver_stage_id"]]):
        fields["requests"] = [msgspec.convert(request, OmniEngineCoreRequest) for request in fields["requests"]]
    fields["sampling_params_list"] = [
        _decode_sampling_params(params, stage_type)
        for params, stage_type in zip(fields["sampling_params_list"], stage_types, strict=True)
    ]
    return NextStageInputMessage(**fields)


def _decode_sampling_params(params: dict[str, Any], stage_type: Literal["llm", "diffusion"]) -> OmniSamplingParams:
    if is_diffusion(stage_type):
        return OmniDiffusionSamplingParams.from_dict(params)
    return msgspec.convert(params, SamplingParams)
