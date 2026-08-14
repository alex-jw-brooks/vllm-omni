# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared decode utilities for Granite ProsodyLM.

NAR preamble layout (used by both the model and the stage input processor)
and CTC greedy decode / interleaving helpers for NLE text normalization.
"""

from __future__ import annotations

import torch

_N_PROSODY_DIMS = 5


def compute_preamble_layout(
    emotion_control: int,
    nar_global_dims: list[int],
    compact_preamble: bool,
) -> tuple[int, list[int]]:
    """Compute the NAR preamble layout (size and dimension indices).

    Returns (preamble_len, dim_indices) where dim_indices maps each preamble
    position to its NAR head index (-1 = not predicted).
    """
    sil_dim = _N_PROSODY_DIMS
    emo_aro_dim = _N_PROSODY_DIMS + 1
    emo_val_dim = _N_PROSODY_DIMS + 2

    if nar_global_dims:
        compact_preamble = True

    if not compact_preamble:
        n_legacy = _N_PROSODY_DIMS + 1
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


def greedy_ctc_decode(
    logits: torch.Tensor,
    blank_id: int,
    copy_id: int | None = None,
    src_at_pos: list[int | None] | None = None,
) -> list[int]:
    """Greedy CTC decode: argmax -> collapse consecutive duplicates -> remove blanks.

    When copy_id is set, COPY emissions resolve to the source token via
    src_at_pos. Adjacent COPYs with different source tokens are NOT collapsed
    (copy-aware de-duplication).
    """
    pred_ids = logits.argmax(dim=-1).tolist()
    decoded: list[int] = []
    prev = None
    prev_copy_src = None
    for t, tid in enumerate(pred_ids):
        is_copy = copy_id is not None and tid == copy_id
        if is_copy:
            src = src_at_pos[t] if src_at_pos is not None else None
            if src is None:
                prev = tid
                continue
            if tid != prev or src != prev_copy_src:
                decoded.append(src)
                prev_copy_src = src
        else:
            if tid != prev and tid != blank_id:
                decoded.append(tid)
            prev_copy_src = None
        prev = tid
    return decoded


def interleave_with_blanks(
    token_ids: list[int],
    blank_id: int,
    slots_per_gap: int = 3,
) -> list[int]:
    """Insert blank_id slots between each token and at both ends."""
    out: list[int] = []
    for tid in token_ids:
        out.extend([blank_id] * slots_per_gap)
        out.append(tid)
    out.extend([blank_id] * slots_per_gap)
    return out
