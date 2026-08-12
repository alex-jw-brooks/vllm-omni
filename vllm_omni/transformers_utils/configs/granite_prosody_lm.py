# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Granite ProsodyLM (TTS via prosody prediction).

Extends GraniteMoeHybridConfig with prosody-specific fields. The backbone
is a standard GraniteMoeHybrid; this config adds the inference knobs for
AR text normalization (Stage 0) and NAR prosody prediction (Stage 1).
"""

from __future__ import annotations

import inspect
import logging

from transformers import AutoConfig, GraniteMoeHybridConfig

logger = logging.getLogger(__name__)


# TODO: remove once config fields / arch are finalized
def _warn_unhandled_kwargs(cls: type, kwargs: dict) -> None:
    """Log a warning for kwargs not in our declared fields or any base class."""
    known = set()
    for ancestor in cls.__mro__:
        if ancestor is object:
            continue
        sig = inspect.signature(ancestor.__init__)
        known.update(p for p in sig.parameters if p not in ("self", "kwargs", "args"))
    unknown = {k for k in kwargs if k not in known and not k.startswith("_")}
    if unknown:
        logger.warning(
            "GraniteProsodyLMConfig: unhandled kwargs (stored but not used at inference): %s",
            sorted(unknown),
        )


class GraniteProsodyLMConfig(GraniteMoeHybridConfig):
    model_type = "granite_prosody_lm"

    def __init__(
        self,
        # Stage routing
        nar_mode: bool = False,
        template_mode: str = "T1",
        compact_preamble: bool = True,
        # Token IDs
        mask_token_id: int = 0,
        num_start_id: int = 0,
        num_end_id: int = 0,
        sil_token_id: int = 0,
        sep1_token_id: int = 0,
        sep2_token_id: int = 0,
        sep_norm_token_id: int = 0,
        sep_f0_token_id: int = 0,
        # NAR decode
        nar_tiers: list | None = None,
        nar_iterations: int = 2,
        nar_iterations_per_tier: list | None = None,
        nar_global_dims: list | None = None,
        nar_ensure_self_attend: bool = True,
        nar_history_causal: bool = True,
        nar_max_history: int = 1,
        nar_temperature: float = 0.2,
        nar_mask_schedule: str = "random",
        # Speaker preamble (pre-tokenized by export script)
        speaker_prefix_ids: list | None = None,
        speaker_suffix_ids: list | None = None,
        default_f0_bin: int = 0,
        # NLE CTC text normalization (NAR text norm via CTC head)
        ctc_text_norm: bool = False,
        ctc_head_vocab_size: int = 0,
        ctc_blank_index: int = 0,
        ctc_editor_copy_op: bool = False,
        ctc_copy_index: int | None = None,
        blank_token_id: int = 0,
        ctc_slots_per_gap: int = 3,
        ctc_steps: int = 1,
        # Grammar processor (AR mode)
        emotion_control: int = 0,
        n_emo_bins: int = 0,
        emo_start_id: int = 0,
        # Validation-only (reject if non-default)
        expert_chain_mode: str | None = None,
        use_fractional_posids: bool = False,
        dedicated_prosody_tokens: bool = False,
        **kwargs,
    ):
        _warn_unhandled_kwargs(type(self), kwargs)
        super().__init__(**kwargs)

        self.nar_mode = nar_mode
        self.template_mode = template_mode
        self.compact_preamble = compact_preamble

        self.mask_token_id = mask_token_id
        self.num_start_id = num_start_id
        self.num_end_id = num_end_id
        self.sil_token_id = sil_token_id
        self.sep1_token_id = sep1_token_id
        self.sep2_token_id = sep2_token_id
        self.sep_norm_token_id = sep_norm_token_id
        self.sep_f0_token_id = sep_f0_token_id

        self.nar_tiers = nar_tiers or []
        self.nar_iterations = nar_iterations
        self.nar_iterations_per_tier = nar_iterations_per_tier or []
        self.nar_global_dims = nar_global_dims or []
        self.nar_ensure_self_attend = nar_ensure_self_attend
        self.nar_history_causal = nar_history_causal
        self.nar_max_history = nar_max_history
        self.nar_temperature = nar_temperature
        self.nar_mask_schedule = nar_mask_schedule

        self.speaker_prefix_ids = speaker_prefix_ids or []
        self.speaker_suffix_ids = speaker_suffix_ids or []
        self.default_f0_bin = default_f0_bin

        self.emotion_control = emotion_control
        self.n_emo_bins = n_emo_bins
        self.emo_start_id = emo_start_id

        self.ctc_text_norm = ctc_text_norm
        self.ctc_head_vocab_size = ctc_head_vocab_size
        self.ctc_blank_index = ctc_blank_index
        self.ctc_editor_copy_op = ctc_editor_copy_op
        self.ctc_copy_index = ctc_copy_index
        self.blank_token_id = blank_token_id
        self.ctc_slots_per_gap = ctc_slots_per_gap
        self.ctc_steps = ctc_steps

        self.expert_chain_mode = expert_chain_mode
        self.use_fractional_posids = use_fractional_posids
        self.dedicated_prosody_tokens = dedicated_prosody_tokens

        # vLLM uses is_causal to select attention type (causal vs bidirectional).
        # NAR needs bidirectional attention; AR needs causal.
        self.is_causal = not nar_mode

        validate_config(self)


def validate_config(config: GraniteProsodyLMConfig) -> None:
    """Reject unsupported features at load time.

    Called from the model's __init__ so misconfiguration crashes early
    rather than producing silent wrong output.
    """
    if config.emotion_control != 0:
        raise ValueError(
            f"emotion_control={config.emotion_control} is not supported. "
            "Only emotion_control=0 (no emotion) is implemented."
        )
    if config.n_emo_bins > 0:
        raise ValueError(f"n_emo_bins={config.n_emo_bins} is not supported. Emotion codebooks are not implemented.")
    if config.expert_chain_mode is not None:
        raise ValueError(
            f"expert_chain_mode={config.expert_chain_mode!r} is not supported. Expert chain decode is not implemented."
        )
    if config.use_fractional_posids:
        raise ValueError("use_fractional_posids=True is not supported. Fractional position IDs are not implemented.")
    if config.template_mode != "T1":
        raise ValueError(
            f"template_mode={config.template_mode!r} is not supported. Only template_mode='T1' is implemented."
        )
    if config.dedicated_prosody_tokens:
        raise ValueError(
            "dedicated_prosody_tokens=True is not supported. Separate prosody token embeddings are not implemented."
        )

    if config.nar_mode:
        if config.mask_token_id == 0:
            raise ValueError("nar_mode=True requires mask_token_id to be set.")
        if config.nar_mask_schedule not in ("random",):
            raise ValueError(
                f"nar_mask_schedule={config.nar_mask_schedule!r} is not "
                "supported. Only 'random' (flat, no tiers) is implemented."
            )
        if config.nar_mask_schedule != "random":
            if not config.nar_tiers:
                raise ValueError("nar_mode=True with tiered schedule requires nar_tiers to be set.")
            if not config.nar_iterations_per_tier:
                raise ValueError("nar_mode=True with tiered schedule requires nar_iterations_per_tier to be set.")


AutoConfig.register("granite_prosody_lm", GraniteProsodyLMConfig)

__all__ = ["GraniteProsodyLMConfig", "validate_config"]
