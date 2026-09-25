# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from typing import Any

from vllm.pooling_params import PoolingParams

from vllm_omni.inputs.data import OmniSamplingParams


def stage_defaults(
    *stages: tuple[type[OmniSamplingParams | PoolingParams], dict[str, Any] | None],
) -> tuple[list[OmniSamplingParams | PoolingParams], list[dict[str, Any] | None]]:
    """Return an engine's ``(default_sampling_params_list, default_sampling_kwargs_list)``.

    Each stage is ``(params_cls, kwargs)``; pooling stages pass ``None`` kwargs.
    """
    params_list = [params_cls(**(kwargs or {})) for params_cls, kwargs in stages]
    return params_list, [None if kwargs is None else dict(kwargs) for _, kwargs in stages]
