# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequestState multimodal DELTA drain and consolidation guard."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.engine.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# NOTE: detokenizer and logprobs aren't really used here, but we mock them since
# some of the utils called in vLLM superclass assert require them to be None.
_DETOK = MagicMock(
    output_token_ids=[0],
    get_next_output_text=MagicMock(return_value=""),
    num_output_tokens=MagicMock(return_value=1),
)
_LOGPROBS = MagicMock(logprobs=None, cumulative_logprob=None, prompt_logprobs=None)

_DEFAULT_STATE_KWARGS = dict(
    request_id="r",
    external_req_id="r",
    parent_req=None,
    request_index=0,
    lora_request=None,
    prompt=None,
    prompt_token_ids=[0],
    prompt_embeds=None,
    logprobs_processor=_LOGPROBS,
    detokenizer=_DETOK,
    max_tokens_param=None,
    arrival_time=0.0,
    queue=None,
    log_stats=False,
    stream_interval=1,
)


def _make_state(output_kind: RequestOutputKind):
    return OmniRequestState(**_DEFAULT_STATE_KWARGS, output_kind=output_kind)


def test_init_empty_dict():
    """Ensure mm_accumulated is initially empty."""
    assert _make_state(RequestOutputKind.CUMULATIVE).mm_accumulated == {}
    assert _make_state(RequestOutputKind.DELTA).mm_accumulated == {}


def test_delta_drains_output_modality_per_step():
    """DELTA drains the mm_type key (output modality) but preserves hidden-state keys."""
    s = _make_state(RequestOutputKind.DELTA)
    audio1, audio2, hs1, hs2 = [torch.ones(num_elem) for num_elem in range(1, 5)]

    # Add audio and hidden state tensors
    s.add_multimodal_tensor(audio1, mm_type="audio")  # should be drained
    s.add_multimodal_tensor(hs1, mm_type="hidden")  # shouldn't be drained

    out1 = s._new_completion_output([1], None, None)
    out1_audio = out1.multimodal_output["audio"]
    out1_hidden = out1.multimodal_output["hidden"]
    assert isinstance(out1_audio, torch.Tensor)
    assert torch.equal(out1.multimodal_output["audio"], audio1)
    assert isinstance(out1_hidden, torch.Tensor)
    assert torch.equal(out1.multimodal_output["hidden"], hs1)

    # After emission, hidden states should remain, but audio is drained
    assert set(s.mm_accumulated.keys()) == {"hidden"}

    s.add_multimodal_tensor(audio2, "audio")
    s.add_multimodal_tensor(hs2, mm_type="hidden")
    out2 = s._new_completion_output([2], None, None)
    out2_audio = out2.multimodal_output["audio"]
    out2_hidden = out2.multimodal_output["hidden"]
    assert isinstance(out2_audio, torch.Tensor)
    assert torch.equal(out2_audio, audio2)
    # Since hidden isn't drained, it's grown to a list
    assert isinstance(out2_hidden, list) and len(out2_hidden) == 2
    assert torch.equal(hs1, out2_hidden[0])
    assert torch.equal(hs2, out2_hidden[1])


def test_cumulative_does_not_drain():
    """Ensure that calling _new_completion_output doesn't clear accumulated mm data."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    t1 = torch.ones(5)
    s.add_multimodal_tensor(t1, "audio")
    s._new_completion_output([1], None, None)
    assert "audio" in s.mm_accumulated
    assert torch.equal(s.mm_accumulated["audio"], t1)


def test_finish_consolidates_non_delta():
    """Ensure tensor consolidation is called for non delta images."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    len_1 = 5
    len_2 = 4
    s.add_multimodal_tensor(torch.ones(len_1), "audio")
    s.add_multimodal_tensor(torch.ones(len_2), "audio")

    result = s.make_request_output([1], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)

    audio = result.outputs[0].multimodal_output["audio"]
    assert isinstance(audio, torch.Tensor) and audio.shape[0] == len_1 + len_2


def test_finish_consolidation_for_hs_delta():
    """Ensure finish doesn't drop the accumulated hidden states."""
    s = _make_state(RequestOutputKind.DELTA)
    # hidden state accumulation (nothing drained)
    s.add_multimodal_tensor({"foo": torch.ones(5, 4)}, mm_type="hidden")
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["foo"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 5

    # Since we don't drain the hidden states, if we add 3 elements, we should get 8
    s.add_multimodal_tensor({"foo": torch.ones(3, 4)}, mm_type="hidden")
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["foo"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 8
    assert "foo" in s.mm_accumulated


def test_finish_consolidation_for_mm_delta():
    """Ensure audio consolidates as expected. Note that for now, audio is
    handled as a special case."""
    s = _make_state(RequestOutputKind.DELTA)
    # multimodal data accumulation (drained)
    s.add_multimodal_tensor({"audio": torch.ones(5, 4)}, mm_type="audio")
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["audio"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 5

    # Since we did drain the hidden states, we no longer get the 5 back
    s.add_multimodal_tensor({"audio": torch.ones(3, 4)}, mm_type="audio")
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["audio"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 3
    assert "audio" not in s.mm_accumulated  # drained
