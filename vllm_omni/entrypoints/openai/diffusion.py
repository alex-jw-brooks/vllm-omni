# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI diffusion-stage request helpers.

These helpers are stage-based rather than image-format-specific."""

from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.sampling import OmniSamplingRequest

MAX_UINT32_SEED = 2**32 - 1


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    sampling_request: OmniSamplingRequest,
    stage_configs: list[Any],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    if not stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )
    sampling_params_list = sampling_request.to_sampling_params_list(
        engine_client.default_sampling_params_list, engine_client.default_sampling_kwargs_list
    )

    async for output in engine_client.generate(
        sampling_params_list=sampling_params_list,
        **kwargs,
    ):
        result = output

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result
