# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for OmniOpenAIServingChat sampling params handling.

Tests that standard OpenAI API parameters (max_tokens, temperature, etc.)
are correctly applied to the comprehension stage while preserving YAML defaults.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams

from tests.helpers.serving_chat import build_serving_chat
from tests.helpers.stage_defaults import stage_defaults
from vllm_omni.entrypoints.openai.protocol.sampling import parse_sampling_params_list
from vllm_omni.entrypoints.openai.sampling_requests import ChatSamplingRequest, DiffusionSamplingRequest
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_comprehension_stage(mocker: MockerFixture):
    """Create a mock comprehension stage with is_comprehension=True."""
    stage = mocker.MagicMock()
    stage.is_comprehension = True
    stage.model_stage = "comprehension"
    return stage


@pytest.fixture
def mock_other_stage(mocker: MockerFixture):
    """Create a mock non-comprehension stage."""
    stage = mocker.MagicMock()
    stage.is_comprehension = False
    stage.model_stage = "other"
    return stage


@pytest.fixture
def default_comprehension_params():
    """Default sampling kwargs for comprehension stage (from YAML)."""
    return {"temperature": 0.4, "top_p": 0.9, "top_k": 1, "max_tokens": 4353, "seed": 42, "repetition_penalty": 1.05}


@pytest.fixture
def default_other_params():
    """Default sampling kwargs for non-comprehension stage (from YAML)."""
    return {"temperature": 0.9, "top_k": 50, "max_tokens": 4096, "seed": 42}


@pytest.fixture
def mock_engine_client(
    mock_comprehension_stage,
    mock_other_stage,
    default_comprehension_params,
    default_other_params,
    mocker: MockerFixture,
):
    """Create mock engine client with stage_configs and default_sampling_params_list."""
    engine_client = mocker.MagicMock()
    engine_client.stage_configs = [mock_comprehension_stage, mock_other_stage]
    engine_client.default_sampling_params_list, engine_client.default_sampling_kwargs_list = stage_defaults(
        (SamplingParams, default_comprehension_params), (SamplingParams, default_other_params)
    )
    return engine_client


def test_serving_boundary_normalizes_declared_root_and_nested_extras(mock_engine_client):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_extra_body_params = frozenset({"cfg_text_scale", "negative_prompt"})
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        modalities=["image"],
        num_inference_steps=7,
        quality="high",
        size="768x512",
        negative_prompt="avoid blur",
        lora={"name": "adapter"},
        cfg_text_scale=7.0,
        extra_body={"extra_args": {"sample_solver": "euler"}},
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {
        "cfg_text_scale": 7.0,
        "sample_solver": "euler",
    }
    assert diffusion_request_args["cfg_text_scale"] == 7.0
    assert diffusion_request_args["num_inference_steps"] == 7
    assert diffusion_request_args["quality"] == "high"
    assert diffusion_request_args["size"] == "768x512"
    assert diffusion_request_args["negative_prompt"] == "avoid blur"
    assert diffusion_request_args["lora"] == {"name": "adapter"}
    assert diffusion_request_args["modalities"] == ["image"]


def test_unknown_root_extra_does_not_claim_canonical_extra(mock_engine_client):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_extra_body_params = frozenset()
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        pipeline_option="ignored-root-value",
        extra_args={"pipeline_option": "canonical-value"},
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {
        "pipeline_option": "canonical-value",
    }
    assert "pipeline_option" not in diffusion_request_args


def test_serving_boundary_rejects_invalid_quality(mock_engine_client):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_extra_body_params = frozenset()
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        modalities=["image"],
        quality="medium",
    )

    with pytest.raises(ValueError, match="quality must be one of"):
        serving_chat._normalize_diffusion_request_args(request)


def test_unregistered_cfg_scale_aliases_common_true_cfg_scale(mock_engine_client):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_extra_body_params = frozenset()
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        cfg_scale=7.0,
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {}
    assert diffusion_request_args == {"true_cfg_scale": 7.0}


@pytest.mark.parametrize(
    ("registered", "request_kwargs"),
    [
        ({"cfg_scale"}, {"cfg_scale": 7.0, "extra_args": {"cfg_scale": 8.0}}),
        (set(), {"cfg_scale": 7.0, "true_cfg_scale": 2.0}),
    ],
    ids=["registered-model-extra", "unregistered-common-alias"],
)
def test_cfg_scale_owner_conflicts_are_rejected(mock_engine_client, registered, request_kwargs):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_extra_body_params = frozenset(registered)
    request = ChatCompletionRequest(model="test", messages=[], **request_kwargs)

    with pytest.raises(ValueError, match="provided more than once"):
        serving_chat._normalize_diffusion_request_args(request)


@pytest.mark.parametrize("diffusion_mode", [True, False], ids=["pure", "mixed"])
def test_duplicate_extras_return_the_same_bad_request_before_dispatch(
    mock_engine_client,
    mocker: MockerFixture,
    diffusion_mode: bool,
):
    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    serving_chat._diffusion_mode = diffusion_mode
    serving_chat._diffusion_extra_body_params = frozenset({"sample_solver"})
    serving_chat.engine_client.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    check_model = mocker.patch.object(serving_chat, "_check_model", new=mocker.AsyncMock())
    pure_dispatch = mocker.patch.object(
        serving_chat,
        "_create_diffusion_chat_completion",
        new=mocker.AsyncMock(),
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        sample_solver="euler",
        extra_args={"sample_solver": "ddim"},
    )

    response = asyncio.run(serving_chat._create_chat_completion(request))

    assert response.error.code == 400
    assert response.error.message == (
        'Diffusion request parameters were provided more than once: "sample_solver": '
        "request.sample_solver, request.extra_args.sample_solver."
    )
    check_model.assert_not_awaited()
    pure_dispatch.assert_not_awaited()


def test_pure_consumer_preserves_defaults_and_separate_cfg_owners(mock_engine_client, mocker: MockerFixture):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    serving_chat = build_serving_chat(engine_client=mock_engine_client)
    captured: dict[str, object] = {}

    async def generate(**kwargs):
        captured.update(kwargs)
        if False:
            yield None

    serving_chat._diffusion_model_name = "test"
    params_list, kwargs_list = stage_defaults(
        (OmniDiffusionSamplingParams, {"extra_args": {"solver": "euler", "stage_default": True}})
    )
    serving_chat._diffusion_engine = SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="diffusion")],
        default_sampling_params_list=params_list,
        default_sampling_kwargs_list=kwargs_list,
        generate=generate,
    )
    serving_chat._diffusion_mode = True
    serving_chat._diffusion_extra_body_params = frozenset({"cfg_scale"})
    serving_chat._extract_diffusion_prompt_and_media = mocker.Mock(return_value=("prompt", [], [], []))
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        modalities=["image"],
        num_inference_steps=7,
        size="768x512",
        negative_prompt="avoid blur",
        cfg_scale=7.0,
        true_cfg_scale=2.0,
        quality="high",
        extra_body={"extra_args": {"solver": "ddim"}},
    )

    response = asyncio.run(serving_chat._create_chat_completion(request))

    (sampling_params,) = captured["sampling_params_list"]
    assert sampling_params.extra_args == {
        "cfg_scale": 7.0,
        "solver": "ddim",
        "stage_default": True,
    }
    assert sampling_params.true_cfg_scale == 2.0
    assert sampling_params.num_inference_steps == 7
    assert sampling_params.quality == "high"
    assert (sampling_params.height, sampling_params.width) == (512, 768)
    assert captured["prompt"]["negative_prompt"] == "avoid blur"
    assert response.error.message == "No output generated from AsyncOmni"


def test_mixed_consumer_keeps_root_common_args_with_nested_extras(mock_engine_client, mocker: MockerFixture):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    mock_engine_client.stage_configs = [
        SimpleNamespace(stage_type="llm", is_comprehension=True),
        SimpleNamespace(stage_type="diffusion", is_comprehension=False),
    ]
    mock_engine_client.default_sampling_params_list, mock_engine_client.default_sampling_kwargs_list = stage_defaults(
        (SamplingParams, {}), (OmniDiffusionSamplingParams, {})
    )
    mock_engine_client.output_modalities = ["image"]
    mock_engine_client.errored = False
    mock_engine_client.renderer = SimpleNamespace(get_tokenizer=lambda: object())

    captured: dict[str, object] = {}

    async def results():
        if False:
            yield None

    def generate(**kwargs):
        captured.update(kwargs)
        return results()

    mock_engine_client.generate = generate
    serving_chat = build_serving_chat(
        engine_client=mock_engine_client,
        models=SimpleNamespace(model_name=lambda _: "test"),
        online_renderer=SimpleNamespace(validate_chat_template=lambda **_: None),
        trust_request_chat_template=True,
    )
    serving_chat._diffusion_mode = False
    serving_chat._diffusion_extra_body_params = frozenset()
    mocker.patch.multiple(
        serving_chat,
        _check_model=mocker.AsyncMock(return_value=None),
        _maybe_get_adapters=mocker.Mock(return_value=None),
        _effective_chat_template_kwargs=mocker.Mock(return_value={}),
        _preprocess_chat=mocker.AsyncMock(return_value=([], [{"prompt": "raw"}])),
        _base_request_id=mocker.Mock(return_value="test"),
        _extract_diffusion_prompt_and_images_from_messages=mocker.Mock(return_value=("prompt", [])),
        _log_inputs=mocker.Mock(),
        chat_completion_full_generator=mocker.AsyncMock(return_value="done"),
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "prompt"}],
        modalities=["image"],
        num_inference_steps=7,
        size="768x512",
        negative_prompt="avoid blur",
        extra_body={"extra_args": {"solver": "euler"}},
    )

    assert asyncio.run(serving_chat._create_chat_completion(request)) == "done"

    sampling_params_list = captured["sampling_params_list"]
    assert sampling_params_list[0].extra_args == {"target_h": 512, "target_w": 768}
    diffusion_params = sampling_params_list[1]
    assert diffusion_params.num_inference_steps == 7
    assert diffusion_params.quality is None
    assert (diffusion_params.height, diffusion_params.width) == (512, 768)
    assert diffusion_params.extra_args == {"solver": "euler"}
    assert captured["prompt"]["negative_prompt"] == "avoid blur"


def test_text_only_request_reaches_engine_with_comprehension_task_mode(mock_engine_client, mocker: MockerFixture):
    """Exercise the production chat path, not only the tagging helper."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    mock_engine_client.stage_configs = [
        SimpleNamespace(stage_type="llm", is_comprehension=True),
        SimpleNamespace(stage_type="diffusion", is_comprehension=False),
    ]
    mock_engine_client.default_sampling_params_list, mock_engine_client.default_sampling_kwargs_list = stage_defaults(
        (SamplingParams, {}), (OmniDiffusionSamplingParams, {})
    )
    mock_engine_client.output_modalities = ["text", "image"]
    mock_engine_client.errored = False
    mock_engine_client.renderer = SimpleNamespace(get_tokenizer=lambda: object())
    captured: dict[str, object] = {}

    async def results():
        if False:
            yield None

    def generate(**kwargs):
        captured.update(kwargs)
        return results()

    mock_engine_client.generate = generate
    serving_chat = build_serving_chat(
        engine_client=mock_engine_client,
        models=SimpleNamespace(model_name=lambda _: "test"),
        online_renderer=SimpleNamespace(validate_chat_template=lambda **_: None),
        trust_request_chat_template=True,
    )
    serving_chat._diffusion_mode = False
    serving_chat._diffusion_extra_body_params = frozenset()
    mocker.patch.multiple(
        serving_chat,
        _check_model=mocker.AsyncMock(return_value=None),
        _maybe_get_adapters=mocker.Mock(return_value=None),
        _effective_chat_template_kwargs=mocker.Mock(return_value={}),
        _preprocess_chat=mocker.AsyncMock(return_value=([], [{"prompt": "raw"}])),
        _base_request_id=mocker.Mock(return_value="test"),
        _log_inputs=mocker.Mock(),
        chat_completion_full_generator=mocker.AsyncMock(return_value="done"),
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "describe this image"}],
        modalities=["text"],
        extra_body={"size": "64x64", "num_inference_steps": 3, "guidance_scale": 9.0},
    )

    assert asyncio.run(serving_chat._create_chat_completion(request)) == "done"

    sampling_params_list = cast(list[Any], captured["sampling_params_list"])
    assert sampling_params_list[0].extra_args == {"ar_task_mode": "comprehension"}
    assert sampling_params_list[1].extra_args == {}
    # Image-generation args do not reach the DiT stage on a text-only request.
    dit_params = sampling_params_list[1]
    assert (dit_params.height, dit_params.num_inference_steps, dit_params.guidance_scale) == (None, None, None)
    assert captured["output_modalities"] == ["text"]


# =============================================================================
# Tests for ChatSamplingRequest (request -> per-stage params)
# =============================================================================


def _chat_sampling_params_list(stage_kwargs: list[dict[str, Any]], **request_kwargs: Any) -> list[Any]:
    """Convert a chat request for AR stages built from ``stage_kwargs``."""
    request = ChatCompletionRequest(model="test", messages=[], **request_kwargs)
    defaults = stage_defaults(*((SamplingParams, kwargs) for kwargs in stage_kwargs))
    return ChatSamplingRequest(request=request, comprehension_stage_id=0).to_sampling_params_list(*defaults)


@pytest.mark.parametrize(
    ("field", "value"),
    # Same-named fields share one generic path; max_tokens has its own.
    [("temperature", 0.8), ("max_tokens", 100)],
)
def test_request_field_overrides_only_comprehension_stage(
    default_comprehension_params, default_other_params, field, value
):
    defaults = [default_comprehension_params, default_other_params]

    result = _chat_sampling_params_list(defaults, **{field: value})

    assert getattr(result[0], field) == value
    assert getattr(result[1], field) == getattr(SamplingParams(**default_other_params), field)
    assert getattr(SamplingParams(**default_comprehension_params), field) != value
    # Unset fields keep their stage values.
    assert result[0].top_k == 1
    assert result[0].repetition_penalty == 1.05


@pytest.mark.parametrize(
    ("field", "value"),
    [("ignore_eos", False), ("presence_penalty", 0.0)],
)
def test_explicit_vllm_default_overrides_stage_value(field, value):
    default = {"ignore_eos": True, "min_tokens": 10, "skip_special_tokens": False, "presence_penalty": 0.5}

    (result,) = _chat_sampling_params_list([default], **{field: value})

    assert getattr(result, field) == value


def test_request_field_feeding_a_renamed_param_keeps_stage_value_when_unchanged():
    # echo only feeds prompt_logprobs; echo=false leaves it as vLLM would.
    (result,) = _chat_sampling_params_list([{"prompt_logprobs": 2}], echo=False)

    assert result.prompt_logprobs == 2


def test_request_without_sampling_fields_overrides_nothing():
    # Includes fields vLLM would otherwise take from the request.
    default = SamplingParams(
        n=2,
        temperature=0.3,
        top_p=0.5,
        top_k=7,
        min_p=0.1,
        repetition_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.4,
        max_tokens=9,
        min_tokens=3,
        seed=5,
        stop_token_ids=[4],
        ignore_eos=True,
        logprobs=2,
        prompt_logprobs=1,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        include_stop_str_in_output=True,
        detokenize=False,
        extra_args={"k": 1},
    )
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])

    assert ChatSamplingRequest(request=request, comprehension_stage_id=0).request_overrides(default) == {}


def test_beam_search_is_rejected():
    with pytest.raises(ValueError, match="Beam search"):
        _chat_sampling_params_list([{}], use_beam_search=True)


@pytest.mark.parametrize("field", ["stop", "stop_token_ids"])
def test_empty_stop_lists_preserve_defaults(field):
    default = {"temperature": 0.5, "stop": ["<|im_end|>"], "stop_token_ids": [2, 3]}

    (result,) = _chat_sampling_params_list([default], **{field: []})

    assert getattr(result, field) == default[field]


def test_declared_extra_args_reach_every_ar_stage_only():
    defaults = stage_defaults((SamplingParams, {}), (SamplingParams, {}), (OmniDiffusionSamplingParams, {}))
    request = ChatCompletionRequest(model="test", messages=[])
    sampling_request = ChatSamplingRequest(
        request=request, comprehension_stage_id=None, declared_extra_args={"cfg_text_scale": 4.0}
    )

    params_list = sampling_request.to_sampling_params_list(*defaults)

    ar_extra_args = {"ar_task_mode": "comprehension", "cfg_text_scale": 4.0}
    assert [params.extra_args for params in params_list] == [ar_extra_args, ar_extra_args, {}]


def test_request_fields_use_vllm_conversion():
    default = {"max_tokens": 64, "seed": 42, "detokenize": False, "stop_token_ids": [2]}

    (params,) = _chat_sampling_params_list(
        [default],
        max_completion_tokens=8,
        watermarking=False,
        stop_token_ids=[100],
        min_p=0.1,
        seed=None,
    )

    assert (params.max_tokens, params.watermarking, params.min_p) == (8, False, 0.1)
    # vLLM merges request stop_token_ids with the server defaults.
    assert params.stop_token_ids == [100, 2]
    # Null and unset fields keep their stage values.
    assert (params.seed, params.detokenize) == (42, False)


def test_multiple_choices_are_rejected_for_multi_stage_pipelines():
    with pytest.raises(ValueError, match="n > 1"):
        ChatSamplingRequest(
            request=ChatCompletionRequest(model="test", messages=[], n=2), comprehension_stage_id=0
        ).to_sampling_params_list(*stage_defaults((SamplingParams, {}), (OmniDiffusionSamplingParams, {})))


def test_multiple_choices_reach_single_stage_pipelines():
    (params,) = _chat_sampling_params_list([{}], n=2)

    assert params.n == 2


@pytest.mark.parametrize(
    ("request_kwargs", "params_field", "expected"),
    [
        ({"max_completion_tokens": 8}, "max_tokens", 8),
        ({"logprobs": True, "top_logprobs": 3}, "logprobs", 3),
        ({"echo": True, "logprobs": True, "top_logprobs": 2}, "prompt_logprobs", 2),
    ],
    ids=["max_completion_tokens", "top_logprobs", "echo"],
)
def test_renamed_request_fields_reach_their_params_field(request_kwargs, params_field, expected):
    # Catches vLLM renaming a request field.
    (params,) = _chat_sampling_params_list([{"max_tokens": 64}], **request_kwargs)

    assert getattr(params, params_field) == expected


def test_response_format_reaches_structured_outputs():
    (params,) = _chat_sampling_params_list([{}], response_format={"type": "json_object"})

    assert params.structured_outputs is not None
    assert params.structured_outputs.json_object is True


def test_unknown_request_extras_do_not_override_stage_fields():
    # ChatCompletionRequest allows extras; a same-named extra is not a request field.
    (params,) = _chat_sampling_params_list([{"detokenize": False}], detokenize=True)

    assert params.detokenize is False


def test_image_generation_fields_win_over_declared_and_request_extra_args():
    request = ChatCompletionRequest(model="test", messages=[], vllm_xargs={"target_h": 1, "src": "request"})
    sampling_request = ChatSamplingRequest(
        request=request,
        comprehension_stage_id=0,
        declared_extra_args={"src": "declared"},
        diffusion=DiffusionSamplingRequest(height=512, width=768),
    )

    (params,) = sampling_request.to_sampling_params_list(*stage_defaults((SamplingParams, {})))

    assert params.extra_args == {"ar_task_mode": "comprehension", "target_h": 512, "target_w": 768, "src": "declared"}


def test_vllm_xargs_merge_over_stage_extra_args():
    default = {"extra_args": {"keep": 1, "replace": 1}}

    (params,) = _chat_sampling_params_list([default], vllm_xargs={"replace": 2, "ar_task_mode": "generation"})

    # Request extras also win over the text-only task marker.
    assert params.extra_args == {"keep": 1, "replace": 2, "ar_task_mode": "generation"}


def test_explicit_openai_fields_override_caller_stage_params():
    defaults = stage_defaults((SamplingParams, {"max_tokens": 64}), (SamplingParams, {"max_tokens": 96}))
    parsed = parse_sampling_params_list([{"temperature": 0.2, "max_tokens": 8}], *defaults)

    params_list = _chat_sampling_params_list(parsed, temperature=0.7)

    assert (params_list[0].temperature, params_list[0].max_tokens) == (0.7, 8)
    assert params_list[1].max_tokens == 96


# =============================================================================
# Tests for _get_comprehension_stage_index
# =============================================================================


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([True, False], 0), ([False, True], 1), ([False, False], None)],
)
def test_get_comprehension_stage_index(flags, expected):
    stage_configs = [SimpleNamespace(is_comprehension=flag) for flag in flags]

    assert OmniOpenAIServingChat._get_comprehension_stage_index(stage_configs) == expected


# =============================================================================
# Tests for _resolve_height_width_from_extra_body
# =============================================================================


class TestResolveHeightWidth:
    def test_explicit_height_width(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"height": 512, "width": 768})
        assert h == 512
        assert w == 768

    def test_size_string(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "768x512"})
        assert w == 768
        assert h == 512

    def test_size_string_uppercase(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "768X512"})
        assert w == 768
        assert h == 512

    def test_size_fallback_when_height_missing(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "512x512", "width": 1024})
        # height is None -> size fallback fires and sets BOTH width and height
        assert h == 512
        assert w == 512

    def test_empty_extra_body(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({})
        assert h is None
        assert w is None

    def test_invalid_size_format_ignored(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "invalid"})
        assert h is None
        assert w is None


# Tests for the text-only ar_task_mode marker (#6088)


def _tagged_params(stages, modalities):
    request = ChatCompletionRequest(model="test", messages=[], modalities=modalities)
    return ChatSamplingRequest(request=request, comprehension_stage_id=None).to_sampling_params_list(
        *stage_defaults(*stages)
    )


@pytest.mark.parametrize("modalities", [None, [], ["text"]])
def test_text_only_chat_tags_every_ar_stage_as_comprehension(modalities):
    params_list = _tagged_params([(SamplingParams, {}), (SamplingParams, {"extra_args": {"custom": 1}})], modalities)

    assert [params.extra_args for params in params_list] == [
        {"ar_task_mode": "comprehension"},
        {"custom": 1, "ar_task_mode": "comprehension"},
    ]


@pytest.mark.parametrize("modalities", [["image"], ["text", "audio"], ["video"]])
def test_non_text_output_request_is_untouched(modalities):
    (params,) = _tagged_params([(SamplingParams, {})], modalities)

    assert params.extra_args is None


def test_request_ar_task_mode_reaches_only_the_comprehension_stage():
    request = ChatCompletionRequest(model="test", messages=[], vllm_xargs={"ar_task_mode": "generation"})
    sampling_request = ChatSamplingRequest(request=request, comprehension_stage_id=0)

    first, second = sampling_request.to_sampling_params_list(
        *stage_defaults((SamplingParams, {}), (SamplingParams, {}))
    )

    assert [first.extra_args, second.extra_args] == [{"ar_task_mode": "generation"}, {"ar_task_mode": "comprehension"}]


def test_deploy_ar_task_mode_is_preserved():
    (params,) = _tagged_params([(SamplingParams, {"extra_args": {"ar_task_mode": "generation"}})], ["text"])

    assert params.extra_args == {"ar_task_mode": "generation"}
