# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request grammar constraint for prosody token generation.

Enforces valid prosody token structure during AR generation:
  [emo?] SIL DUR (NUM*K SIL DUR)*N SEP2

Two modes:
  - Fixed:   num_words known → pre-built position-indexed pattern
  - Dynamic: num_words unknown → state machine enforces structure,
             model chooses when to emit SEP2
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class ProsodyGrammarProcessor:
    """Per-request grammar constraint for prosody token generation.

    When ``num_words`` is provided, builds a position-indexed pattern of
    allowed token sets.  When ``num_words`` is None, runs a state machine
    that enforces structure without knowing the total word count.
    """

    FEATURE_NAMES = ["dur", "prs_1", "prs_2", "prs_3", "prs_4"]

    def __init__(
        self,
        num_words: int | None,
        num_start_id: int,
        num_end_id: int,
        sil_token_id: int,
        sep2_token_id: int,
        k: int = 5,
        emotion_control: int = 0,
        emo_start_id: int | None = None,
        n_emo_bins: int = 65,
        feature_ranges: dict[str, tuple[int, int]] | None = None,
    ):
        self.num_words = num_words
        self.k = k
        self.num_start_id = num_start_id
        self.num_end_id = num_end_id
        self.sil_token_id = sil_token_id
        self.sep2_token_id = sep2_token_id
        self.emotion_control = emotion_control
        self.emo_start_id = emo_start_id
        self.n_emo_bins = n_emo_bins
        self.feature_ranges = feature_ranges
        self.pos = 0

        self._num_ids = set(range(num_start_id, num_end_id))
        self._sil_set = frozenset({sil_token_id})
        self._sep2_set = frozenset({sep2_token_id})

        if num_words is not None:
            self.pattern: list[set[int]] = self._build_pattern()
            self._use_pattern = True
        else:
            self.pattern = []
            self._use_pattern = False
            self._state = "emo" if emotion_control > 0 else "sil"
            self._emo_remaining = emotion_control
            self._word_feat_idx = 0

    def _build_pattern(self) -> list[set[int]]:
        sil_set = {self.sil_token_id}
        sep2_set = {self.sep2_token_id}

        if self.feature_ranges is not None:
            feat_sets = [set(range(*self.feature_ranges[name])) for name in self.FEATURE_NAMES]
            dur_ids = feat_sets[0]
            word_sets = feat_sets
        else:
            num_ids = self._num_ids
            dur_ids = num_ids
            word_sets = [num_ids] * self.k

        pattern: list[set[int]] = []

        if self.emotion_control > 0 and self.emo_start_id is not None:
            arousal_set = set(range(self.emo_start_id, self.emo_start_id + self.n_emo_bins))
            pattern.append(arousal_set)
            if self.emotion_control > 1:
                valence_set = set(
                    range(
                        self.emo_start_id + self.n_emo_bins,
                        self.emo_start_id + 2 * self.n_emo_bins,
                    )
                )
                pattern.append(valence_set)

        pattern.append(sil_set)
        pattern.append(dur_ids)

        for _ in range(self.num_words):
            for s in word_sets:
                pattern.append(s)
            pattern.append(sil_set)
            pattern.append(dur_ids)

        pattern.append(sep2_set)
        return pattern

    def _dynamic_allowed(self) -> set[int]:
        """State machine for dynamic mode (num_words unknown)."""
        if self._state == "emo":
            if self.emo_start_id is not None:
                if self._emo_remaining > 1:
                    return set(range(self.emo_start_id, self.emo_start_id + self.n_emo_bins))
                else:
                    return set(
                        range(
                            self.emo_start_id + self.n_emo_bins,
                            self.emo_start_id + 2 * self.n_emo_bins,
                        )
                    )
            self._state = "sil"
            return self._dynamic_allowed()

        if self._state == "sil":
            return set(self._sil_set)

        if self._state == "dur":
            return set(self._num_ids)

        if self._state == "word_feat":
            return set(self._num_ids)

        if self._state == "word_boundary":
            return set(self._sil_set) | set(self._sep2_set)

        return set(self._num_ids) | set(self._sil_set) | set(self._sep2_set)

    def _dynamic_advance(self, token_id: int) -> None:
        """Advance state machine after emitting a token."""
        if self._state == "emo":
            self._emo_remaining -= 1
            if self._emo_remaining <= 0:
                self._state = "sil"
        elif self._state == "sil":
            self._state = "dur"
        elif self._state == "dur":
            self._word_feat_idx = 0
            self._state = "word_feat"
        elif self._state == "word_feat":
            self._word_feat_idx += 1
            if self._word_feat_idx >= self.k:
                self._state = "word_boundary"
        elif self._state == "word_boundary":
            if token_id == self.sil_token_id:
                self._state = "dur"
            # if sep2, generation should stop via stop_token_ids

    def __call__(
        self,
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_pattern:
            if self.pos < len(self.pattern):
                allowed = self.pattern[self.pos]
                mask = torch.full_like(logits, float("-inf"))
                valid = [i for i in allowed if i < logits.shape[-1]]
                mask[valid] = 0.0
                logits = logits + mask
            self.pos += 1
        else:
            if output_token_ids:
                self._dynamic_advance(output_token_ids[-1])
            allowed = self._dynamic_allowed()
            mask = torch.full_like(logits, float("-inf"))
            valid = [i for i in allowed if i < logits.shape[-1]]
            mask[valid] = 0.0
            logits = logits + mask

        return logits
