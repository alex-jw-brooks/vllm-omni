# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ensure that the `output_kind` can be correctly coerced by AsyncOmni
for multistage pipelines."""

from __future__ import annotations

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("skip_clone", [True, False])
def test_cumulative_default_becomes_delta(skip_clone):
    """Ensure cumulative messages are coercible to delta."""
    sp = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
    sp.skip_clone = skip_clone
    result = AsyncOmni.coerce_cumulative_messages([sp])[0]
    assert isinstance(result, SamplingParams)
    assert result.output_kind == RequestOutputKind.DELTA
    assert (skip_clone and sp is result) or (not skip_clone and sp is not result)


@pytest.mark.parametrize("output_kind", [(RequestOutputKind.DELTA), RequestOutputKind.FINAL_ONLY])
def test_non_cumulative_are_preserved(output_kind):
    """Ensure messages that are not cumulative are preserved."""
    sp = SamplingParams(output_kind=output_kind)
    result = AsyncOmni.coerce_cumulative_messages([sp])[0]
    assert isinstance(result, SamplingParams)
    assert result.output_kind == output_kind
