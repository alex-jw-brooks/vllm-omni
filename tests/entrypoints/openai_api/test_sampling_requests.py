# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct tests for request -> per-stage sampling params conversion."""

import dataclasses

import pytest
from vllm import SamplingParams

from tests.helpers.stage_defaults import stage_defaults
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateAudioGenerateRequest, OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest, VideoParams
from vllm_omni.entrypoints.openai.sampling_requests import (
    _VIDEO_DIFFUSION_PARAMS,
    _VIDEO_EXTRA_ARG_FIELDS,
    ARDiffusionSamplingRequest,
    DiffusionSamplingRequest,
    VideoSamplingRequest,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_speech_request_maps_extra_params_and_seed_to_first_ar_stage():
    defaults = stage_defaults(
        (SamplingParams, {"temperature": 0.9, "extra_args": {"keep": 1}}), (SamplingParams, {"temperature": 0.5})
    )
    request = OpenAICreateSpeechRequest(input="hi", seed=7, extra_params={"temperature": 0.2, "top_k": 5, "x": "y"})

    stage0, stage1 = request.to_sampling_params_list(*defaults)

    assert (stage0.temperature, stage0.top_k, stage0.seed) == (0.2, 5, 7)
    assert stage0.extra_args == {"keep": 1, "temperature": 0.2, "top_k": 5, "x": "y", "tts_local_seed": 7}
    assert (stage1.temperature, stage1.seed, stage1.extra_args) == (0.5, None, None)
    assert defaults[1][0]["extra_args"] == {"keep": 1}


def test_speech_request_maps_extra_params_and_seed_to_diffusion_tts_stage():
    defaults = stage_defaults((OmniDiffusionSamplingParams, {"num_inference_steps": 12, "extra_args": {"keep": 1}}))
    request = OpenAICreateSpeechRequest(
        input="hi", seed=7, extra_params={"num_inference_steps": "4", "guidance_scale": 2}
    )

    (params,) = request.to_sampling_params_list(*defaults)

    # The step scheduler reads the top-level fields; the pipeline reads extra_args.
    assert (params.num_inference_steps, params.guidance_scale) == (4, 2.0)
    assert params.extra_args == {"keep": 1, "num_inference_steps": "4", "guidance_scale": 2, "seed": 7}
    assert defaults[1][0]["num_inference_steps"] == 12


@pytest.mark.parametrize("field", ["num_inference_steps", "guidance_scale"])
def test_speech_request_rejects_non_numeric_diffusion_overrides(field):
    request = OpenAICreateSpeechRequest(input="hi", extra_params={field: "fast"})

    with pytest.raises(ValueError, match=field):
        request.to_sampling_params_list(*stage_defaults((OmniDiffusionSamplingParams, {})))


def test_speech_request_rejects_non_numeric_sampling_overrides():
    request = OpenAICreateSpeechRequest(input="hi", extra_params={"temperature": "hot"})

    with pytest.raises(ValueError, match="temperature must be a number"):
        request.to_sampling_params_list(*stage_defaults((SamplingParams, {})))


def test_audio_generate_request_overrides_diffusion_defaults():
    defaults = stage_defaults(
        (
            OmniDiffusionSamplingParams,
            {"num_inference_steps": 50, "num_outputs_per_prompt": 2, "extra_args": {"keep": 1}},
        )
    )
    request = OpenAICreateAudioGenerateRequest(
        input="rain", seed=42, guidance_scale=3.0, audio_length=5.0, audio_start=1.0
    )

    (params,) = request.to_sampling_params_list(*defaults)

    assert (params.num_inference_steps, params.guidance_scale, params.num_outputs_per_prompt) == (50, 3.0, 1)
    assert params.extra_args == {"keep": 1, "audio_start_in_s": 1.0, "audio_end_in_s": 6.0}
    # The diffusion worker builds the generator from the seed on its device.
    assert (params.seed, params.generator) == (42, None)


def test_diffusion_request_targets_ar_and_dit_stages():
    defaults = stage_defaults(
        (SamplingParams, {"seed": 1}),
        (SamplingParams, {}),
        (OmniDiffusionSamplingParams, {"guidance_scale": 4.0, "extra_args": {"keep": 1}}),
    )
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    request = ARDiffusionSamplingRequest(
        diffusion=DiffusionSamplingRequest(
            height=512,
            width=768,
            seed=9,
            strength=0.3,
            extra_args={"cfg_text_scale": 2.0},
            lora_request=lora_request,
            lora_scale=0.5,
        ),
        comprehension_stage_id=0,
        stop_token_ids=[5],
    )

    comprehension, other_ar, dit = request.to_sampling_params_list(*defaults)

    assert (comprehension.seed, comprehension.stop_token_ids) == (9, [5])
    assert comprehension.extra_args == {"target_h": 512, "target_w": 768}
    assert (other_ar.seed, other_ar.stop_token_ids, other_ar.extra_args) == (None, [5], None)
    assert (dit.height, dit.width, dit.seed, dit.strength) == (512, 768, 9, 0.3)
    assert dit.guidance_scale == 4.0
    assert dit.extra_args == {"keep": 1, "cfg_text_scale": 2.0}
    assert (dit.lora_request, dit.lora_scale) == (lora_request, 0.5)
    assert defaults[1][2]["extra_args"] == {"keep": 1}


def test_diffusion_request_from_extra_body_takes_known_fields():
    request = DiffusionSamplingRequest.from_extra_body(
        {"num_inference_steps": 4, "guidance_scale": 2.0, "bot_task": "think"},
        guidance_scale=3.0,
    )

    assert (request.num_inference_steps, request.guidance_scale) == (4, 3.0)


@pytest.mark.parametrize("name", ["extra_args", "lora_request", "lora_scale", "height_not_provided"])
def test_diffusion_request_from_extra_body_ignores_handler_fields(name):
    request = DiffusionSamplingRequest.from_extra_body({name: 1})

    assert getattr(request, name) is None


def test_diffusion_request_from_extra_body_keeps_body_value_over_null_kwarg():
    request = DiffusionSamplingRequest.from_extra_body({"seed": 5}, seed=None)

    assert request.seed == 5


def test_diffusion_request_fields_are_diffusion_params():
    # The dataclass fields are the allowlist; each must name a real params field.
    request_fields = {f.name for f in dataclasses.fields(DiffusionSamplingRequest)}

    assert request_fields <= set(OmniDiffusionSamplingParams.__dataclass_fields__)


def test_video_request_maps_each_diffusion_stage_from_its_own_default():
    defaults = stage_defaults(
        (SamplingParams, {"temperature": 0.3}),
        (OmniDiffusionSamplingParams, {"fps": 16, "guidance_scale": 2.0}),
        (OmniDiffusionSamplingParams, {"fps": 16, "guidance_scale": 5.0}),
    )
    request = VideoGenerationRequest(prompt="p", size="640x360", num_frames=9, seed=3, extra_params={"k": "v"})
    sampling_request = VideoSamplingRequest(request=request, video_params=request.resolve_video_params())

    ar, first, second = sampling_request.to_sampling_params_list(*defaults)

    assert ar.temperature == 0.3
    for params, guidance_scale in ((first, 2.0), (second, 5.0)):
        assert (params.width, params.height, params.num_frames, params.seed) == (640, 360, 9, 3)
        assert params.guidance_scale == guidance_scale
        assert params.fps == 16
        assert params.extra_args == {"k": "v"}


def test_video_request_applies_handler_resolved_values():
    request = VideoGenerationRequest(prompt="p", num_inference_steps=None)
    sampling_request = VideoSamplingRequest(
        request=request,
        video_params=request.resolve_video_params(default_fps=24.0, default_num_frames=81),
        fps=24.0,
        default_num_inference_steps=40,
        duration=5.0,
        emit_request_lifecycle=True,
    )

    first, second = sampling_request.to_sampling_params_list(
        *stage_defaults((OmniDiffusionSamplingParams, {}), (OmniDiffusionSamplingParams, {"num_inference_steps": 8}))
    )

    assert (first.num_inference_steps, first.num_frames, first.fps, first.frame_rate) == (40, 81, 24.0, 24.0)
    assert second.num_inference_steps == 8
    assert first.extra_args == {"duration": 5.0}
    assert first.emit_request_lifecycle is True


def test_video_request_takes_size_only_from_resolved_params():
    defaults = stage_defaults((OmniDiffusionSamplingParams, {"width": 320, "height": 240, "num_outputs_per_prompt": 2}))
    request = VideoGenerationRequest(prompt="p", width=640)
    video_params = request.resolve_video_params().model_copy(update={"width": 640, "height": None})
    sampling_request = VideoSamplingRequest(request=request, video_params=video_params)

    (params,) = sampling_request.to_sampling_params_list(*defaults)

    # A half-resolved size keeps the default; unset fields keep deploy defaults.
    assert (params.width, params.height, params.num_outputs_per_prompt) == (320, 240, 2)


def test_video_request_lora_keeps_default_scale_when_unset():
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    request = VideoGenerationRequest(prompt="p")
    sampling_request = VideoSamplingRequest(
        request=request, video_params=request.resolve_video_params(), lora_request=lora_request
    )

    (params,) = sampling_request.to_sampling_params_list(
        *stage_defaults((OmniDiffusionSamplingParams, {"lora_scale": 0.7}))
    )

    assert (params.lora_request, params.lora_scale) == (lora_request, 0.7)


@pytest.mark.parametrize("name", _VIDEO_EXTRA_ARG_FIELDS)
def test_video_extra_arg_fields_are_request_fields(name):
    assert name in VideoGenerationRequest.model_fields


def test_video_allowlist_covers_every_request_field_named_like_a_diffusion_param():
    # A new protocol field named like a params field must be allowed or excluded on purpose.
    same_named = VideoGenerationRequest.model_fields.keys() & OmniDiffusionSamplingParams.__dataclass_fields__.keys()

    assert same_named - VideoParams.model_fields.keys() == set(_VIDEO_DIFFUSION_PARAMS)
