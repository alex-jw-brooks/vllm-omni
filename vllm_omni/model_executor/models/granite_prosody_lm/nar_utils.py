# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for Granite ProsodyLM NAR decode.

Functions here are imported by both the model class and the stage input
processor, avoiding a circular dependency between them.
"""

from __future__ import annotations

_N_PROSODY_DIMS = 5


def compute_preamble_layout(
    emotion_control: int,
    nar_global_dims: list[int],
    compact_preamble: bool,
    n_prosody_dims: int = _N_PROSODY_DIMS,
) -> tuple[int, list[int]]:
    """Compute the NAR preamble layout (size and dimension indices).

    Returns (preamble_len, dim_indices) where dim_indices maps each preamble
    position to its NAR head index (-1 = not predicted).

    Mirrors the reference compute_preamble_layout from nar_collator.py.
    """
    sil_dim = n_prosody_dims
    emo_aro_dim = n_prosody_dims + 1
    emo_val_dim = n_prosody_dims + 2

    if nar_global_dims:
        compact_preamble = True

    if not compact_preamble:
        n_legacy = n_prosody_dims + 1
        dims = [
            emo_aro_dim if emotion_control > 0 else -1,
            emo_val_dim if emotion_control > 1 else -1,
        ]
        dims.extend([-1] * (n_legacy - 3))
        dims.append(sil_dim)
        return n_legacy, dims

    dims: list[int] = []
    if emotion_control > 0:
        dims.append(emo_aro_dim)
    if emotion_control > 1:
        dims.append(emo_val_dim)
    for d in nar_global_dims:
        dims.append(d)
    dims.append(sil_dim)
    return len(dims), dims
