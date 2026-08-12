"""CTC utilities for NLE text normalization.

Greedy CTC decode and interleaving helpers, ported from the reference
implementation (llm/ctc_decode.py, nle_textnorm.py).
"""

from __future__ import annotations

import torch


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
