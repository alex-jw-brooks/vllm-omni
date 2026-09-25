# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-stage sampling params for requests whose mapping needs handler context."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from pydantic import BaseModel
from typing_extensions import Self
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import PoolingParams

from vllm_omni.entrypoints.openai.protocol.sampling import (
    OmniSamplingRequest,
    SamplingOverrides,
    merge_sampling_overrides,
)
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest, VideoParams
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams
from vllm_omni.lora.request import LoRARequest

# Fields set by handlers; ``from_extra_body`` never reads them from the client.
_RESOLVED = {"resolved": True}


def _provided_fields(request: BaseModel) -> dict[str, Any]:
    """Return the declared request fields the caller set to a non-null value."""
    declared = type(request).model_fields
    return {
        name: value
        for name in request.model_fields_set
        if name in declared and (value := getattr(request, name)) is not None
    }


@dataclass(frozen=True, kw_only=True)
class DiffusionSamplingRequest(OmniSamplingRequest):
    """Diffusion params an image request may set, applied over the stage defaults.

    The fields are an allowlist: params such as ``latents`` or ``generator``
    must not be settable by clients.
    """

    height: int | None = None
    width: int | None = None
    height_not_provided: bool | None = field(default=None, metadata=_RESOLVED)
    width_not_provided: bool | None = field(default=None, metadata=_RESOLVED)
    seed: int | None = None
    generator_device: str | None = None
    num_outputs_per_prompt: int | None = None
    quality: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None
    true_cfg_scale: float | None = None
    strength: float | None = None
    num_frames: int | None = None
    layers: int | None = None
    resolution: int | None = None
    extra_args: Mapping[str, object] | None = field(default=None, metadata=_RESOLVED)
    lora_request: LoRARequest | None = field(default=None, metadata=_RESOLVED)
    lora_scale: float | None = field(default=None, metadata=_RESOLVED)

    @classmethod
    def from_extra_body(cls, extra_body: Mapping[str, Any], **kwargs: Any) -> Self:
        """Build from an ``extra_body``; non-null ``kwargs`` take precedence."""
        body_fields = {
            f.name: extra_body[f.name] for f in fields(cls) if f.name in extra_body and not f.metadata.get("resolved")
        }
        return cls(**{**body_fields, **{name: value for name, value in kwargs.items() if value is not None}})

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


def _comprehension_image_overrides(diffusion: DiffusionSamplingRequest) -> dict[str, object]:
    """Seed and target size for the comprehension stage of an image request."""
    overrides: dict[str, object] = {}
    if diffusion.seed is not None:
        overrides["seed"] = diffusion.seed
    if diffusion.height is not None and diffusion.width is not None:
        # e.g. GLM-Image pre-computes M-RoPE positions from the target grid.
        overrides["extra_args"] = {"target_h": int(diffusion.height), "target_w": int(diffusion.width)}
    return overrides


@dataclass(frozen=True, kw_only=True)
class ARDiffusionSamplingRequest(OmniSamplingRequest):
    """Image generation through AR+DiT stages."""

    diffusion: DiffusionSamplingRequest
    comprehension_stage_id: int | None = None
    stop_token_ids: list[int] | None = None

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        overrides: dict[str, object] = {}
        if self.stop_token_ids is not None:
            overrides["stop_token_ids"] = self.stop_token_ids
        if stage_id == self.comprehension_stage_id:
            overrides.update(_comprehension_image_overrides(self.diffusion))
        return overrides

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        return self.diffusion.diffusion_overrides(stage_id, default)


# ``to_sampling_params`` builds params through ``from_optional``, so its
# parameters are the fields a request can set.
_REQUEST_SETTABLE_FIELDS = tuple(inspect.signature(SamplingParams.from_optional).parameters)


def _request_settable_fields(params: SamplingParams) -> dict[str, Any]:
    return {name: getattr(params, name) for name in _REQUEST_SETTABLE_FIELDS}


@functools.cache
def _unset_chat_request(request_cls: type[ChatCompletionRequest], stream: bool | None) -> ChatCompletionRequest:
    """Return a request with no sampling fields set.

    ``stream`` is kept so ``output_kind`` matches the real request.
    """
    return request_cls.model_construct(stream=stream)


@dataclass(frozen=True, kw_only=True)
class ChatSamplingRequest(OmniSamplingRequest):
    """Chat completion request, optionally generating images through AR+DiT stages.

    Request sampling fields apply to the comprehension stage; declared extras
    apply to every AR stage. Image fields win over declared extras, which win
    over request fields.
    """

    request: ChatCompletionRequest
    comprehension_stage_id: int | None
    diffusion: DiffusionSamplingRequest = field(default_factory=DiffusionSamplingRequest)
    declared_extra_args: Mapping[str, object] = field(default_factory=dict)

    def to_sampling_params_list(
        self,
        default_sampling_params_list: Sequence[OmniSamplingParams | PoolingParams],
        default_sampling_kwargs_list: Sequence[Mapping[str, Any] | None],
    ) -> list[OmniSamplingParams | PoolingParams]:
        if self.request.use_beam_search:
            raise ValueError("Beam search is not supported.")
        # Downstream stages consume only the first comprehension output.
        if (self.request.n or 1) > 1 and len(default_sampling_params_list) > 1:
            raise ValueError("n > 1 is not supported for multi-stage pipelines.")
        return super().to_sampling_params_list(default_sampling_params_list, default_sampling_kwargs_list)

    def request_overrides(self, default: SamplingParams) -> dict[str, object]:
        """Return the SamplingParams fields the request sets over ``default``.

        The request is converted by vLLM. A field applies if the caller set it
        explicitly, or if it differs from converting an unset request; the
        latter covers renamed fields such as ``max_completion_tokens``.
        """
        request = self.request
        max_tokens = request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens
        if max_tokens is None:
            max_tokens = default.max_tokens
        unset_request = _unset_chat_request(type(request), request.stream)
        server_defaults = _request_settable_fields(default)
        try:
            requested = _request_settable_fields(request.to_sampling_params(max_tokens, server_defaults))
            unset = _request_settable_fields(unset_request.to_sampling_params(default.max_tokens, server_defaults))
        except VLLMValidationError as exc:
            raise ValueError(f"Invalid sampling params: {exc}") from exc
        explicit = {name for name, value in _provided_fields(request).items() if value != []}
        return {name: value for name, value in requested.items() if name in explicit or value != unset[name]}

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        task_mode: dict[str, object] = {}
        # Keeps generation models (e.g. HunyuanImage3) from emitting image
        # scaffold tokens into text answers (#6088).
        text_only = not set(getattr(self.request, "modalities", None) or []) - {"text"}
        if text_only and "ar_task_mode" not in (default.extra_args or {}):
            task_mode = {"extra_args": {"ar_task_mode": "comprehension"}}
        declared = {"extra_args": self.declared_extra_args} if self.declared_extra_args else {}
        if stage_id != self.comprehension_stage_id:
            return merge_sampling_overrides(task_mode, declared)
        return merge_sampling_overrides(
            task_mode,
            self.request_overrides(default),
            declared,
            _comprehension_image_overrides(self.diffusion),
        )

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        return self.diffusion.diffusion_overrides(stage_id, default)


# Video request fields copied to the same-named diffusion param, and fields
# passed through ``extra_args``. Size, frames, and fps come from the handler.
_VIDEO_DIFFUSION_PARAMS = (
    "num_outputs_per_prompt",
    "seed",
    "quality",
    "num_inference_steps",
    "guidance_scale",
    "guidance_scale_2",
    "true_cfg_scale",
    "boundary_ratio",
    "enable_frame_interpolation",
    "frame_interpolation_exp",
    "frame_interpolation_scale",
    "frame_interpolation_model_path",
)
_VIDEO_EXTRA_ARG_FIELDS = (
    "aspect_ratio",
    "short_edge",
    "start_time_seconds",
    "flow_shift",
    "generate_sound",
    "sound_duration",
)


@dataclass(frozen=True, kw_only=True)
class VideoSamplingRequest(OmniSamplingRequest):
    """Video request plus the handler's resolved size, frames, fps, and model defaults."""

    request: VideoGenerationRequest
    video_params: VideoParams
    fps: float | None = None
    default_num_inference_steps: int | None = None
    duration: float | None = None
    lora_request: LoRARequest | None = None
    lora_scale: float | None = None
    emit_request_lifecycle: bool = False

    def sampling_overrides(self, stage_id: int, default: SamplingParams) -> SamplingOverrides:
        # e.g. minimax_h3's AR stage keeps its deploy defaults.
        return {}

    def diffusion_overrides(self, stage_id: int, default: OmniDiffusionSamplingParams) -> SamplingOverrides:
        provided = _provided_fields(self.request)
        overrides: dict[str, object] = {name: provided[name] for name in _VIDEO_DIFFUSION_PARAMS if name in provided}
        overrides["emit_request_lifecycle"] = self.emit_request_lifecycle
        if "num_inference_steps" not in overrides and default.num_inference_steps is None:
            if self.default_num_inference_steps is not None:
                overrides["num_inference_steps"] = self.default_num_inference_steps
        vp = self.video_params
        if vp.width is not None and vp.height is not None:
            overrides["width"] = vp.width
            overrides["height"] = vp.height
        if vp.num_frames is not None:
            overrides["num_frames"] = vp.num_frames
        if self.fps is not None:
            overrides["fps"] = self.fps
            overrides["frame_rate"] = float(self.fps)
        if self.lora_request is not None:
            overrides["lora_request"] = self.lora_request
            if self.lora_scale is not None:
                overrides["lora_scale"] = self.lora_scale

        extra_args = {name: provided[name] for name in _VIDEO_EXTRA_ARG_FIELDS if name in provided}
        if self.duration is not None and "duration" not in (default.extra_args or {}):
            extra_args["duration"] = self.duration
        overrides["extra_args"] = {**extra_args, **(self.request.extra_params or {})}
        return overrides
