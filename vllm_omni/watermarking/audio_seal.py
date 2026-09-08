# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torchaudio.functional as audio_functional
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

from vllm_omni.watermarking.base import AudioWatermarkerBase
from vllm_omni.watermarking.types import AudioTensor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from audioseal.models import AudioSealDetector, AudioSealWM

try:
    import audioseal.loader as loader
except ImportError:
    loader = PlaceholderModule("audioseal")  # type: ignore[assignment]

_AUDIOSEAL_SAMPLE_RATE = 16_000


@dataclass
class _AudioSealState:
    """Hold state retained between chunks of one AudioSeal stream."""

    batch_size: int
    sample_rate: int
    message: torch.Tensor
    streaming_state: dict[str, object] | None = None


class AudioSealWatermarker(AudioWatermarkerBase[_AudioSealState]):
    """Watermark audio with Meta AudioSeal."""

    supports_stereo = False

    def __init__(self) -> None:
        """Load the streaming AudioSeal generator."""
        super().__init__()
        if isinstance(loader, PlaceholderModule):
            raise ImportError("Audio watermarking requires `pip install 'vllm-omni[watermarking]'`")
        logger.info("Loading AudioSeal watermark generator on CPU")
        with torch.random.fork_rng(devices=[]):
            self._model: AudioSealWM = loader.AudioSeal.load_generator(
                "audioseal_wm_streaming",
                device=torch.device("cpu"),
            )
        self._model.eval()
        self._detector: AudioSealDetector | None = None

    def _new_audio_state(self, data: AudioTensor) -> _AudioSealState:
        """Create a deterministic message for an AudioSeal stream."""
        self._validate_audioseal_input(data)
        batch_size = data.samples.shape[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            message = self._model.random_message(batch_size)
        logger.debug("Started AudioSeal stream with batch size %d", batch_size)
        return _AudioSealState(batch_size, data.sample_rate, message)

    def _watermark_audio(self, data: AudioTensor, state: _AudioSealState) -> AudioTensor:
        """Watermark one chunk using retained AudioSeal state."""
        self._validate_audioseal_input(data)
        if data.samples.shape[0] != state.batch_size or data.sample_rate != state.sample_rate:
            raise ValueError("audio batch size and sample rate must remain stable within a stream")
        if data.samples.shape[-1] == 0:
            return data

        original_device = data.samples.device
        original_dtype = data.samples.dtype
        source = data.samples.to(device="cpu", dtype=torch.float32)
        # AudioSeal requires 16k sampling rate, so resample if needed
        model_input = self.maybe_resample(source, data.sample_rate, _AUDIOSEAL_SAMPLE_RATE)

        # We always run AudioSeal in streaming mode for now, so that we don't have to
        # have separate paths for delta and cumulative semantics. Initial benchmarks for
        # full vs streamed AudioSeal don't look compelling enough to have a split path for now.
        with self._model.streaming(state.batch_size):
            if state.streaming_state is not None:
                self._model.encoder.set_streaming_state(state.streaming_state)  # type: ignore[attr-defined]
            watermarked = self._model(
                model_input,
                sample_rate=_AUDIOSEAL_SAMPLE_RATE,
                message=state.message,
            )
            state.streaming_state = self._model.encoder.get_streaming_state()  # type: ignore[attr-defined]

        # if we resampled, we need to resample just the watermark & add it onto the residual.
        # Note that torchaudio.functional.resample can change the length, so we may need to pad.
        if data.sample_rate != _AUDIOSEAL_SAMPLE_RATE:
            residual = self.maybe_resample(watermarked - model_input, _AUDIOSEAL_SAMPLE_RATE, data.sample_rate)
            watermarked = source + self.trim_to_length(residual, source.shape[-1])

        return AudioTensor(
            samples=watermarked.to(device=original_device, dtype=original_dtype),
            sample_rate=data.sample_rate,
        )

    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        """Detect an AudioSeal watermark in audio."""
        self._validate_audioseal_input(data)
        if data.samples.shape[-1] == 0:
            return False
        detector = self._detector
        if detector is None:
            logger.info("Loading AudioSeal watermark detector on CPU")
            with torch.random.fork_rng(devices=[]):
                detector = loader.AudioSeal.load_detector(
                    "audioseal_detector_streaming",
                    device=torch.device("cpu"),
                )
            detector.eval()
            self._detector = detector
        samples = self.maybe_resample(
            data.samples.to(device="cpu", dtype=torch.float32),
            data.sample_rate,
            _AUDIOSEAL_SAMPLE_RATE,
        )
        probability, _ = detector.detect_watermark(
            samples,
            sample_rate=_AUDIOSEAL_SAMPLE_RATE,
        )
        return torch.as_tensor(probability).min().item() > 0.5

    def _validate_audioseal_input(self, data: AudioTensor) -> None:
        """Validate AudioSeal channel and sample-rate requirements."""
        if data.samples.shape[1] != 1:
            raise ValueError("AudioSeal implementation requires mono audio")

    @staticmethod
    def maybe_resample(samples: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        """Resample audio for AudioSeal while preserving leading dimensions."""
        if source_rate == target_rate:
            return samples
        return audio_functional.resample(samples, source_rate, target_rate)

    @staticmethod
    def trim_to_length(samples: torch.Tensor, length: int) -> torch.Tensor:
        """Trim resampling round-up without hiding a short residual."""
        if samples.shape[-1] < length:
            raise RuntimeError("resampled AudioSeal residual is shorter than source audio")
        return samples[..., :length]

    def _close_audio_state(self, state: _AudioSealState) -> None:
        """Discard retained AudioSeal streaming state."""
        state.streaming_state = None
        logger.debug("Closed AudioSeal stream")
