# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import soundfile
import torch
from vllm.utils.import_utils import PlaceholderModule

from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor, audio_seal

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_missing_audioseal_names_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing AudioSeal reports the required install extra."""
    monkeypatch.setattr(audio_seal, "loader", PlaceholderModule("audioseal"))

    with pytest.raises(ImportError, match=r"vllm-omni\[watermarking\]"):
        AudioSealWatermarker()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_rejects_unsupported_channel_layout() -> None:
    """Ensure AudioSeal rejects unsupported channel layouts."""
    pytest.importorskip("audioseal")
    watermarker = AudioSealWatermarker()

    with pytest.raises(ValueError, match="channel dimension must be axis 1"):
        watermarker.watermark(
            "request",
            AudioTensor(torch.zeros((1, 100, 2)), audio_seal._AUDIOSEAL_SAMPLE_RATE),
        )

    watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.parametrize(
    ("channels", "chunk_samples", "noise_std"),
    [
        pytest.param(1, None, 0.05, id="mono-whole"),
        pytest.param(1, 8_000, 0.05, id="mono-streaming"),
        pytest.param(2, 8_000, 0.05, id="stereo-streaming"),
        pytest.param(2, 8_000, 0.95, id="stereo-streaming-loud"),
    ],
)
def test_audioseal_survives_pcm16_wav_encoding(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    channels: int,
    chunk_samples: int | None,
    noise_std: float,
) -> None:
    """Ensure mono and stereo watermarks survive PCM16 quantization."""
    pytest.importorskip("audioseal")
    sample_rate = audio_seal._AUDIOSEAL_SAMPLE_RATE
    generator = torch.Generator().manual_seed(0)
    source = (torch.randn((1, channels, sample_rate * 2), generator=generator) * noise_std).clamp_(-1, 1)
    if noise_std == 0.95:
        assert (source.abs() == 1).any()
    watermarker = AudioSealWatermarker()
    rng_state = torch.random.get_rng_state()
    chunk_samples = chunk_samples or source.shape[-1]

    chunks = [
        watermarker.watermark("request", AudioTensor(chunk, sample_rate)).samples
        for chunk in source.split(chunk_samples, dim=-1)
    ]
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert ("does not support stereo" in caplog.text) == (channels == 2)

    path = tmp_path / f"watermarked-{sample_rate}hz-{channels}ch.wav"
    watermarked = torch.cat(chunks, dim=-1).squeeze(0)
    assert watermarked.shape == source.squeeze(0).shape
    assert watermarked.abs().amax() <= 1
    if channels == 2:
        torch.testing.assert_close(
            watermarked[0] - watermarked[1],
            source[0, 0] - source[0, 1],
        )
    soundfile.write(path, watermarked.T.cpu().numpy(), sample_rate, subtype="PCM_16")
    assert soundfile.info(path).subtype == "PCM_16"
    encoded, encoded_rate = soundfile.read(path, dtype="float32", always_2d=True)

    assert encoded_rate == sample_rate
    reloaded = torch.from_numpy(encoded.T.copy()).unsqueeze(0)
    assert watermarker.is_watermarked(AudioTensor(reloaded, encoded_rate))
    watermarker.finish_request("request")
    watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_watermarks_24khz_stream() -> None:
    """Ensure 24 kHz chunks remain 24 kHz and carry a detectable watermark."""
    pytest.importorskip("audioseal")
    sample_rate = 24_000
    source = (
        torch.randn(
            (1, 1, sample_rate * 2),
            generator=torch.Generator().manual_seed(0),
        )
        * 0.05
    )
    watermarker = AudioSealWatermarker()

    try:
        chunks = [
            watermarker.watermark("request", AudioTensor(chunk, sample_rate))
            for chunk in source.split(sample_rate // 2, dim=-1)
        ]
        watermarked = AudioTensor(
            torch.cat([chunk.samples for chunk in chunks], dim=-1),
            sample_rate,
        )

        assert watermarked.samples.shape == source.shape
        assert all(chunk.sample_rate == sample_rate for chunk in chunks)
        assert watermarker.is_watermarked(watermarked)
    finally:
        watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_is_deterministic_across_interleaved_requests() -> None:
    """Ensure interleaved requests remain isolated and deterministic."""
    pytest.importorskip("audioseal")
    sample_rate = audio_seal._AUDIOSEAL_SAMPLE_RATE
    sources = {
        request_id: torch.randn((1, 1, sample_rate), generator=torch.Generator().manual_seed(seed)) * 0.05
        for request_id, seed in (("request-1", 1), ("request-2", 2))
    }

    rng_state = torch.random.get_rng_state()
    watermarker = AudioSealWatermarker()
    chunks = {request_id: source.split(sample_rate // 2, dim=-1) for request_id, source in sources.items()}
    expected = {}
    for request_id, request_chunks in chunks.items():
        expected[request_id] = torch.cat(
            [
                watermarker.watermark(f"expected-{request_id}", AudioTensor(chunk, sample_rate)).samples
                for chunk in request_chunks
            ],
            dim=-1,
        )
        watermarker.finish_request(f"expected-{request_id}")

    interleaved: dict[str, list[torch.Tensor]] = {"request-1": [], "request-2": []}
    for chunk_index in range(len(next(iter(chunks.values())))):
        for request_id in interleaved:
            data = AudioTensor(chunks[request_id][chunk_index], sample_rate)
            interleaved[request_id].append(watermarker.watermark(request_id, data).samples)

    assert all(torch.equal(torch.cat(interleaved[request_id], dim=-1), expected[request_id]) for request_id in sources)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    watermarker.close()
