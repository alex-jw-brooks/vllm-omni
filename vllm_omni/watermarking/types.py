# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AudioTensor:
    """Normalized float audio shaped `[batch, channels, samples]`."""

    samples: torch.Tensor
    sample_rate: int
