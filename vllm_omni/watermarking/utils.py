# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.watermarking.base import Watermarker, WatermarkException
from vllm_omni.watermarking.converters import (
    MEDIA_CONVERTERS,
    infer_visual_channel_axis,
    media_to_tensor,
    restore_media,
)

logger = init_logger(__name__)

OutputPayload = MultimodalPayload | dict[str, object]


def watermark_media(
    request_id: str,
    modality: OutputModalityNames,
    watermarker: Watermarker,
    data: object,
    metadata: Mapping[str, object],
) -> object:
    """Watermark one media value and restore its output type.

    This is needed since current watermarkers assume torch tensors as inputs.
    """
    try:
        if modality is OutputModalityNames.AUDIO and isinstance(data, list):
            raise TypeError("audio output lists are not supported")
        converter = MEDIA_CONVERTERS[modality]
        source = data
        wrapped_batch = False
        if (
            modality in {OutputModalityNames.IMAGE, OutputModalityNames.VIDEO}
            and isinstance(data, list)
            and len(data) == 1
        ):
            first = data[0]
            expected_rank = 4 if modality is OutputModalityNames.IMAGE else 5
            if isinstance(first, torch.Tensor | np.ndarray) and first.ndim == expected_rank:
                source = first
                wrapped_batch = True
        tensor = media_to_tensor(source, converter.to_tensor)
        if modality in {OutputModalityNames.IMAGE, OutputModalityNames.VIDEO}:
            channel_axis = metadata.get("channel_axis")
            if channel_axis is None:
                channel_axis = infer_visual_channel_axis(source, tensor, modality)
            metadata = {
                **metadata,
                "channel_axis": channel_axis,
            }
        watermarked = watermarker.watermark_output(request_id, tensor, metadata)
        result = restore_media(watermarked, source, converter.restore)
        return [result] if wrapped_batch else result
    except (RuntimeError, TypeError, ValueError) as error:
        raise WatermarkException(f"invalid {modality.value} output") from error


def watermark_payload(
    request_id: str,
    modality: OutputModalityNames,
    watermarker: Watermarker,
    payload: OutputPayload,
) -> None:
    """Watermark one modality payload in place."""
    modality_key = modality.value
    data = payload.get(modality_key)
    if data is None or isinstance(data, list) and not data:
        return
    metadata: Mapping[str, object] = payload
    if modality is OutputModalityNames.AUDIO and payload.get("sr") is None:
        metadata = {"sr": payload.get("audio_sample_rate")}
    result = watermark_media(request_id, modality, watermarker, data, metadata)
    if not isinstance(payload, MultimodalPayload):
        payload[modality_key] = result
    elif modality_key not in payload.tensors:
        payload.metadata[modality_key] = result
    elif isinstance(result, torch.Tensor):
        payload.tensors[modality_key] = result
    else:
        raise WatermarkException("tensor payload must remain a tensor")


def _watermark_core_output(
    output: OmniEngineCoreOutput,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    """watermark engine core outputs."""
    if output.multimodal_output is None:
        return
    for modality_key, watermarker in watermarkers.items():
        modality = OutputModalityNames(modality_key)
        payload = MultimodalPayload.from_raw(output.multimodal_output, modality.value)
        if payload is None:
            continue
        watermark_payload(output.request_id, modality, watermarker, payload)
        output.multimodal_output = payload  # type: ignore[assignment]


def _watermark_visual_media(
    output: OmniRequestOutput,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    """Apply watermarkers (if applicable) to visual tensors."""
    watermarker = watermarkers.get(output.final_output_type)
    if watermarker is None:
        return
    modality = OutputModalityNames(output.final_output_type)
    metadata = output.multimodal_output
    if not isinstance(metadata, Mapping):
        metadata = {}
    output.images = watermark_media(  # type: ignore[assignment]
        output.request_id,
        modality,
        watermarker,
        output.images,
        metadata,
    )


def _watermark_request_output(
    output: RequestOutput,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    payloads: list[OutputPayload] = [
        completion.multimodal_output
        for completion in output.outputs
        if isinstance(completion, MultimodalCompletionOutput) and completion.multimodal_output is not None
    ]

    if isinstance(output, OmniRequestOutput) and not output.outputs:
        if isinstance(output.multimodal_output, dict):
            payloads.append(output.multimodal_output)

    for payload in payloads:
        for modality_key, watermarker in watermarkers.items():
            watermark_payload(
                output.request_id,
                OutputModalityNames(modality_key),
                watermarker,
                payload,
            )

    if isinstance(output, OmniRequestOutput) and not output.outputs and output.images:
        _watermark_visual_media(output, watermarkers)
    if output.finished:
        for watermarker in watermarkers.values():
            watermarker.discard_request_state(output.request_id)


def _watermark_output(
    output: object,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    """Apply watermark to either core or request outputs."""
    if isinstance(output, OmniEngineCoreOutput):
        _watermark_core_output(output, watermarkers)
    elif isinstance(output, RequestOutput):
        _watermark_request_output(output, watermarkers)
    else:
        raise TypeError(f"unsupported output type: {type(output).__name__}")


def _handle_watermark_failure(
    output: object,
    watermarkers: Mapping[str, Watermarker],
    error: WatermarkException,
    is_strict: bool,
) -> None:
    """Handle watermark failures by discarding any active state for this request,
    e.g., streaming buffers. If we're operating in strict mode, reraise the error.
    Otherwise log at error level.
    """
    request_id = getattr(output, "request_id", "unknown")
    for watermarker in watermarkers.values():
        watermarker.discard_request_state(request_id)
    modalities = ", ".join(watermarkers)
    message = f"Failed to watermark {modalities} output for request {request_id}"
    if is_strict:
        raise WatermarkException(message) from error
    logger.exception(message)


def watermark_outputs(
    outputs: Sequence[object],
    watermarkers: Mapping[str, Watermarker],
    is_strict: bool = False,
) -> Sequence[object]:
    """Watermark outputs in place, preserving backend failures unless strict."""
    for output in outputs:
        try:
            # TODO: Ensure failure behavior is correct for when we are handling multiple modalities
            _watermark_output(output, watermarkers)
        except WatermarkException as error:
            _handle_watermark_failure(output, watermarkers, error, is_strict)
    return outputs
