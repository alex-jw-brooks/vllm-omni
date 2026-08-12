# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor: text_norm (Stage 0) → prosody (Stage 1).

Transforms the completed text normalization output into the prosody
stage input. Handles both AR and NAR modes:
  AR  — chat-template-wrapped prompt ending at [SEP_F0]; model generates
        prosody tokens autoregressively.
  NAR — same prompt + masked annotation block; model predicts all mask
        positions via iterative refinement.

The prompt is built from scratch using the normalized text (not forwarded
from Stage 0), matching the reference first-utterance path:
  apply_chat_template([{"role": "assistant", "content":
      norm_text[SEP_NORM]Speaker<|num_tk_F0|>:[SEP_F0]}])
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.model_executor.models.granite_prosody_lm.nar_utils import (
    compute_preamble_layout,
)

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )


def _get_hf_config(transfer_manager: Any):
    model_config = transfer_manager._get_model_config()
    return model_config.hf_config


@lru_cache(maxsize=1)
def _load_base_config(base_dir: str) -> dict:
    config_path = os.path.join(base_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def _is_pipeline_nar_mode(transfer_manager: Any) -> bool:
    """Check whether the pipeline uses NAR prosody mode.

    Reads from the base model directory's config.json (the parent of
    per-stage subdirectories), which carries the pipeline-level nar_mode.
    Stage 0's own config always has nar_mode=False since it's AR.
    """
    model_config = transfer_manager._get_model_config()
    base_dir = model_config.tokenizer or os.path.dirname(model_config.model)
    return _load_base_config(base_dir).get("nar_mode", False)


def _get_tokenizer(transfer_manager: Any):
    model_config = transfer_manager._get_model_config()
    return _load_tokenizer(model_config.tokenizer or model_config.model)


def _extract_normalized_text(
    tokenizer,
    output_token_ids: list[int],
    sep_norm_id: int,
) -> str:
    """Decode the normalized text from Stage 0 output tokens.

    Strips trailing [SEP_NORM] before decoding.
    """
    norm_ids = output_token_ids
    if norm_ids and norm_ids[-1] == sep_norm_id:
        norm_ids = norm_ids[:-1]
    return tokenizer.decode(norm_ids, skip_special_tokens=False)


def _build_prosody_prompt_ids(
    tokenizer,
    normalized_text: str,
    f0_bin: int,
) -> list[int]:
    """Build chat-template-wrapped prosody prompt token IDs.

    Constructs the string:
      norm_text[SEP_NORM]Speaker<|num_tk_F0|>:[SEP_F0]
    wraps it in a chat template as the assistant role, strips trailing EOS,
    and encodes to token IDs.
    """
    f0_token_str = f"<|num_tk_{f0_bin}|>"
    sent_content = f"{normalized_text}[SEP_NORM]Speaker{f0_token_str}:[SEP_F0]"
    chat_msg = [{"role": "assistant", "content": sent_content}]
    full_text = tokenizer.apply_chat_template(
        chat_msg,
        tokenize=False,
    )
    eos = tokenizer.eos_token or ""
    full_text = full_text.rstrip()
    if eos and full_text.endswith(eos):
        full_text = full_text[: -len(eos)]
    return tokenizer.encode(full_text, add_special_tokens=False)


def _build_nar_annotation_block(config: Any, num_words: int) -> list[int]:
    """Build the masked NAR annotation block: [SIL] + preamble + word_cols + [SEP_2].

    All predictable positions are filled with mask_token_id. The caller
    appends this to the prompt token IDs.
    """
    mask_id = config.mask_token_id
    pad_id = config.pad_token_id

    g_pre, preamble_dims = compute_preamble_layout(
        config.emotion_control,
        list(config.nar_global_dims),
        config.compact_preamble,
    )

    preamble = []
    for d in preamble_dims:
        if d == 6:
            preamble.append(mask_id if config.emotion_control > 0 else pad_id)
        elif d == 7:
            preamble.append(mask_id if config.emotion_control > 1 else pad_id)
        elif d == -1:
            preamble.append(pad_id)
        else:
            preamble.append(mask_id)

    word_cols = [mask_id] * (6 * num_words)

    return [config.sil_token_id] + preamble + word_cols + [config.sep2_token_id]


def text_norm_to_prosody(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Sync stage input processor: Stage 0 (text_norm) → Stage 1 (prosody).

    Builds a fresh Stage 1 prompt from Stage 0's normalized text output:
      chat_template(norm_text[SEP_NORM]Speaker<|num_tk_F0|>:[SEP_F0])
    For NAR mode, also appends the masked annotation block.
    """
    config = _get_hf_config(transfer_manager)
    tokenizer = _get_tokenizer(transfer_manager)

    req_id = getattr(request, "req_id", None)

    output_ids = list(getattr(request, "output_token_ids", None) or [])
    output_ids = [t for t in output_ids if t >= 0]

    logger.info(
        "text_norm_to_prosody: req=%s output_len=%d last5=%s",
        req_id,
        len(output_ids),
        output_ids[-5:] if output_ids else [],
    )

    if not output_ids:
        logger.warning(
            "text_norm_to_prosody: empty output_token_ids for req=%s",
            req_id,
        )
        return None

    normalized_text = _extract_normalized_text(
        tokenizer,
        output_ids,
        config.sep_norm_token_id,
    )

    f0_bin = config.default_f0_bin
    prompt_ids = _build_prosody_prompt_ids(tokenizer, normalized_text, f0_bin)

    nar_mode = _is_pipeline_nar_mode(transfer_manager)
    if nar_mode:
        num_words = len(normalized_text.split())
        annot_block = _build_nar_annotation_block(config, num_words)
        prompt_ids = list(prompt_ids) + annot_block
        logger.debug(
            "text_norm_to_prosody (NAR): req=%s, num_words=%d, prompt_len=%d, annot_block_len=%d",
            getattr(request, "req_id", None),
            num_words,
            len(prompt_ids),
            len(annot_block),
        )
    else:
        logger.debug(
            "text_norm_to_prosody (AR): req=%s, prompt_len=%d",
            getattr(request, "req_id", None),
            len(prompt_ids),
        )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(prompt_ids, dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(True, dtype=torch.bool),
            next_stage_prompt_len=len(prompt_ids),
        ),
    )
