# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for AsyncOmni input streaming parameter validation."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)


class TestRealtimeWatermarking:
    @pytest.fixture
    def connection(self, realtime_conn: RealtimeConnection, mocker: MockerFixture) -> RealtimeConnection:
        mocker.patch.object(VllmRealtimeConnection, "handle_event", mocker.AsyncMock())
        return realtime_conn

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("updates", "expected"),
        [
            ([{"watermarking": False}], False),
            # Each session.update replaces the session state, so omitting the field resets it
            ([{"watermarking": False}, {}], True),
        ],
    )
    async def test_session_update_sets_watermarking(
        self, connection: RealtimeConnection, updates: list[dict[str, bool]], expected: bool
    ) -> None:
        for update in updates:
            await connection.handle_event({"type": "session.update", "model": "m", **update})

        assert connection._watermarking is expected
