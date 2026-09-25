# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the per-stage params core: hooks, override merging, and caller-provided lists."""

import pytest
from vllm import SamplingParams
from vllm.pooling_params import PoolingParams

from tests.helpers.stage_defaults import stage_defaults
from vllm_omni.entrypoints.openai.protocol.sampling import (
    OmniSamplingRequest,
    SamplingOverrides,
    build_sampling_params_list,
    merge_sampling_overrides,
    parse_sampling_params_list,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingRequest(OmniSamplingRequest):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        self.calls.append(("llm", stage_id))
        return {"temperature": 0.0}

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        self.calls.append(("diffusion", stage_id))
        return {"extra_args": {"stage": stage_id}}


def test_build_sampling_params_list_builds_each_stage_from_its_default_kwargs():
    defaults = stage_defaults(
        (SamplingParams, {"temperature": 0.2}),
        (OmniDiffusionSamplingParams, {"extra_args": {"k": []}}),
        (PoolingParams, None),
    )

    params_list = build_sampling_params_list(*defaults)

    assert [type(params) for params in params_list] == [SamplingParams, OmniDiffusionSamplingParams, PoolingParams]
    assert all(params is not default for params, default in zip(params_list, defaults[0]))
    assert params_list[0].temperature == 0.2
    params_list[1].extra_args["k"].append("request")
    assert defaults[1][1] == {"extra_args": {"k": []}}


def test_to_sampling_params_list_dispatches_hooks_by_stage_params_type():
    defaults = stage_defaults(
        (SamplingParams, {"temperature": 0.7}),
        (OmniDiffusionSamplingParams, {}),
        (PoolingParams, None),
        (OmniDiffusionSamplingParams, {}),
    )
    request = _RecordingRequest()

    params_list = request.to_sampling_params_list(*defaults)

    assert request.calls == [("llm", 0), ("diffusion", 1), ("diffusion", 3)]
    assert params_list[0].temperature == 0.0
    assert [params_list[1].extra_args, params_list[3].extra_args] == [{"stage": 1}, {"stage": 3}]
    assert defaults[1][0] == {"temperature": 0.7}


class _OverridesRequest(OmniSamplingRequest):
    def __init__(self, **overrides: object) -> None:
        self.overrides = overrides

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        return self.overrides


def _override(default_kwargs: dict[str, object], **overrides: object) -> SamplingParams:
    (params,) = _OverridesRequest(**overrides).to_sampling_params_list(
        *stage_defaults((SamplingParams, default_kwargs))
    )
    return params


def test_request_without_a_hook_for_a_stage_type_fails():
    # A request never silently ignores a stage type it was not written for.
    with pytest.raises(NotImplementedError, match="does not handle diffusion stages"):
        _OverridesRequest().to_sampling_params_list(*stage_defaults((OmniDiffusionSamplingParams, {})))


def test_invalid_override_values_are_client_errors():
    with pytest.raises(ValueError, match="Invalid sampling params"):
        _override({}, presence_penalty=5.0)


def test_unknown_override_field_is_a_server_error():
    # A misspelled hook field is our bug, so it must not surface as a 400.
    with pytest.raises(TypeError):
        _override({}, temperature=0.5)


def test_overrides_build_params_as_if_constructed_directly():
    default_kwargs = {"stop": ["LONGER_STOP"], "stop_token_ids": [7], "temperature": 0.0}
    overrides = {"stop": ["END"], "stop_token_ids": [9], "temperature": 0.7, "include_stop_str_in_output": True}

    params = _override(default_kwargs, **overrides)
    fresh = SamplingParams(**{**default_kwargs, **overrides})

    assert {name: getattr(params, name) for name in SamplingParams.__struct_fields__} == {
        name: getattr(fresh, name) for name in SamplingParams.__struct_fields__
    }


def test_greedy_default_keeps_deploy_top_k_when_request_raises_temperature():
    # vLLM resets top_k on a greedy object; the deploy kwargs still hold it.
    params = _override({"temperature": 0.0, "top_k": 50}, temperature=0.7)

    assert (params.temperature, params.top_k) == (0.7, 50)


def test_to_sampling_params_list_merges_extra_args_over_stage_default():
    default_kwargs = {"temperature": 0.3, "extra_args": {"keep": 1, "replace": 1}}

    params = _override(default_kwargs, extra_args={"replace": 2})

    assert params.extra_args == {"keep": 1, "replace": 2}
    assert params.temperature == 0.3
    assert default_kwargs["extra_args"] == {"keep": 1, "replace": 1}


def test_merge_sampling_overrides_later_layers_win_and_extra_args_merge():
    merged = merge_sampling_overrides(
        {"seed": 1, "extra_args": None},
        {"extra_args": {"a": 1, "b": 1}},
        {"seed": 2, "extra_args": {"b": 2}},
    )

    assert merged == {"seed": 2, "extra_args": {"a": 1, "b": 2}}


def test_parse_sampling_params_list_replaces_given_stages_and_keeps_omitted_tail():
    # AURA exposes three semantic models but runs four engine stages.
    defaults = stage_defaults(
        (SamplingParams, {"max_tokens": 10}),
        (OmniDiffusionSamplingParams, {"num_inference_steps": 50}),
        (SamplingParams, {"max_tokens": 30}),
        (SamplingParams, {"max_tokens": 40}),
    )

    parsed = parse_sampling_params_list([{"max_tokens": 1}, {"num_inference_steps": 4}, {}], *defaults)

    assert parsed == [{"max_tokens": 1}, {"num_inference_steps": 4}, {}, {"max_tokens": 40}]


def test_parse_sampling_params_list_accepts_pooling_stage_dicts():
    parsed = parse_sampling_params_list([{}], *stage_defaults((PoolingParams, None)))

    (params,) = build_sampling_params_list([PoolingParams()], parsed)
    assert type(params) is PoolingParams


def test_parse_sampling_params_list_rejects_invalid_entries():
    defaults = stage_defaults((OmniDiffusionSamplingParams, {}))

    with pytest.raises(ValueError, match="Invalid sampling params for stage 0"):
        parse_sampling_params_list([SamplingParams()], *defaults)
    with pytest.raises(ValueError, match="at most 1 sampling params"):
        parse_sampling_params_list([{}, {}], *defaults)
    with pytest.raises(ValueError, match="Invalid sampling params for stage 0"):
        parse_sampling_params_list([{"not_a_field": 1}], *defaults)
    with pytest.raises(ValueError, match="Invalid sampling params for stage 0"):
        parse_sampling_params_list([{"temperature": -1.0}], *stage_defaults((SamplingParams, {})))
    with pytest.raises(ValueError, match="must be a list"):
        parse_sampling_params_list(None, *defaults)
