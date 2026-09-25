# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-stage sampling params for multi-stage requests."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import PoolingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

# Keyword overrides for one stage's params.
SamplingOverrides: TypeAlias = Mapping[str, Any]


def merge_sampling_overrides(*layers: SamplingOverrides) -> dict[str, Any]:
    """Merge override layers. Later layers win; ``extra_args`` are merged."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for name, value in layer.items():
            if name == "extra_args":
                value = {**(merged.get("extra_args") or {}), **(value or {})}
            merged[name] = value
    return merged


def _build_params(
    params_cls: type[OmniSamplingParams | PoolingParams], kwargs: Mapping[str, Any]
) -> OmniSamplingParams | PoolingParams:
    try:
        return params_cls(**copy.deepcopy(dict(kwargs)))
    except VLLMValidationError as exc:
        # Omni handlers map ValueError, not VLLMValidationError, to 400.
        raise ValueError(f"Invalid sampling params: {exc}") from exc


def build_sampling_params_list(
    default_sampling_params_list: Sequence[OmniSamplingParams | PoolingParams],
    default_sampling_kwargs_list: Sequence[Mapping[str, Any] | None],
) -> list[OmniSamplingParams | PoolingParams]:
    """Build each stage's params from its default kwargs, with no request overrides."""
    return [
        # Pooling stages have no kwargs and take no overrides.
        default.clone() if kwargs is None else _build_params(type(default), kwargs)
        for default, kwargs in zip(default_sampling_params_list, default_sampling_kwargs_list, strict=True)
    ]


class OmniSamplingRequest:
    """Multi-stage analogue of vLLM's ``request.to_sampling_params``.

    As in vLLM, params are built once from the stage's default kwargs plus the
    request's overrides. Subclasses implement the hook for each stage type
    their pipelines contain.
    """

    def to_sampling_params_list(
        self,
        default_sampling_params_list: Sequence[OmniSamplingParams | PoolingParams],
        default_sampling_kwargs_list: Sequence[Mapping[str, Any] | None],
    ) -> list[OmniSamplingParams | PoolingParams]:
        params_list = build_sampling_params_list(default_sampling_params_list, default_sampling_kwargs_list)
        for stage_id, (params, kwargs) in enumerate(zip(params_list, default_sampling_kwargs_list)):
            if isinstance(params, OmniDiffusionSamplingParams):
                overrides = self.diffusion_overrides(stage_id, params)
            elif isinstance(params, SamplingParams):
                overrides = self.sampling_overrides(stage_id, params)
            else:
                continue
            if overrides:
                params_list[stage_id] = _build_params(type(params), merge_sampling_overrides(kwargs, overrides))
        return params_list

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        raise NotImplementedError(f"{type(self).__name__} does not handle LLM stages")

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        raise NotImplementedError(f"{type(self).__name__} does not handle diffusion stages")


def parse_sampling_params_list(
    sampling_params_list: Sequence[Mapping[str, Any]],
    default_sampling_params_list: Sequence[OmniSamplingParams | PoolingParams],
    default_sampling_kwargs_list: Sequence[Mapping[str, Any] | None],
) -> list[Mapping[str, Any] | None]:
    """Convert a caller-provided ``sampling_params_list`` to per-stage kwargs.

    Each dict replaces the stage's default kwargs. Stages past the end of the
    list keep their defaults.
    """
    if isinstance(sampling_params_list, str) or not isinstance(sampling_params_list, Sequence):
        raise ValueError(f"sampling_params_list must be a list, got {type(sampling_params_list).__name__}")
    if len(sampling_params_list) > len(default_sampling_params_list):
        raise ValueError(
            f"Expected at most {len(default_sampling_params_list)} sampling params, got {len(sampling_params_list)}"
        )
    parsed: list[Mapping[str, Any] | None] = []
    for stage_id, (params, default) in enumerate(zip(sampling_params_list, default_sampling_params_list)):
        if not isinstance(params, Mapping):
            raise ValueError(f"Invalid sampling params for stage {stage_id}: {params}")
        try:
            type(default)(**params)
        except (TypeError, VLLMValidationError) as exc:
            raise ValueError(f"Invalid sampling params for stage {stage_id}: {exc}") from exc
        parsed.append(dict(params))
    parsed.extend(default_sampling_kwargs_list[len(parsed) :])
    return parsed
