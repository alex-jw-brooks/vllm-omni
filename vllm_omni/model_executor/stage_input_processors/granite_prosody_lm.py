# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for the Granite ProsodyLM pipeline.

process_text_norm_to_prosody (Stage 0 → Stage 1):
  Orchestrator-path processor. Extracts CTC-decoded tokens from Stage 0's
  multimodal output, builds prosody prompt with NAR annotation block.

process_prosody_to_tts (Stage 1 → Stage 2):
  Orchestrator-path processor. Extracts prosody tokens, builds TTS payload
  (phonemes, prsinf, boundaries, speaker embedding) packed into prompt_token_ids.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.granite_prosody_lm.decode_utils import (
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


@lru_cache(maxsize=1)
def _load_hf_config(model_path: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


@lru_cache(maxsize=1)
def _get_orchestrator_config_and_tokenizer() -> tuple:
    """Load config and tokenizer for orchestrator-path functions.

    Uses GRANITE_PROSODY_LM_MODEL_PATH env var (the pipeline model root),
    or discovers it from the Omni engine's model config.
    """
    model_root = os.environ.get("GRANITE_PROSODY_LM_MODEL_PATH")
    if model_root is None:
        raise RuntimeError("GRANITE_PROSODY_LM_MODEL_PATH must be set for orchestrator-path stage input processing.")
    config = _load_hf_config(os.path.join(model_root, "stage1_prosody"))
    tokenizer = _load_tokenizer(model_root)
    return config, tokenizer


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


def _extract_multimodal_output(source_output: Any) -> dict[str, Any] | None:
    """Extract multimodal output from a source output object.

    Handles both OmniRequestOutput (has .multimodal_output property) and raw
    RequestOutput (multimodal data on .outputs[0].multimodal_output).
    """
    from collections.abc import Mapping

    mm = getattr(source_output, "multimodal_output", None)
    if mm is not None and (isinstance(mm, dict) or isinstance(mm, Mapping)):
        if hasattr(mm, "to_dict"):
            return mm.to_dict()
        return dict(mm) if not isinstance(mm, dict) else mm

    for comp_output in getattr(source_output, "outputs", []):
        mm = getattr(comp_output, "multimodal_output", None)
        if mm is not None:
            if hasattr(mm, "to_dict"):
                return mm.to_dict()
            if isinstance(mm, (dict, Mapping)):
                return dict(mm) if not isinstance(mm, dict) else mm

    return None


def process_text_norm_to_prosody(
    source_outputs: list,
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list:
    """Orchestrator-path processor: Stage 0 (text_norm) → Stage 1 (prosody).

    Called by the orchestrator's process_engine_inputs for LLM_GENERATION stages.
    Extracts CTC-decoded tokens from source_output.multimodal_output, builds
    prosody prompt with NAR annotation block, returns as OmniTokensPrompt.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    results = []
    for source_output in source_outputs:
        mm_output = _extract_multimodal_output(source_output)

        text_norm_tensor = mm_output.get("text_norm_tokens") if mm_output else None
        if text_norm_tensor is None:
            logger.warning(
                "process_text_norm_to_prosody: no text_norm_tokens for req=%s, source_type=%s, mm_keys=%s",
                getattr(source_output, "request_id", "?"),
                type(source_output).__name__,
                list(mm_output.keys()) if mm_output else "None",
            )
            continue

        if isinstance(text_norm_tensor, torch.Tensor):
            output_ids = text_norm_tensor.tolist()
        elif isinstance(text_norm_tensor, list):
            output_ids = text_norm_tensor
        else:
            output_ids = list(text_norm_tensor)

        if not output_ids:
            logger.warning(
                "process_text_norm_to_prosody: empty text_norm_tokens for req=%s",
                getattr(source_output, "request_id", "?"),
            )
            continue

        config, tokenizer = _get_orchestrator_config_and_tokenizer()

        normalized_text = _extract_normalized_text(
            tokenizer,
            output_ids,
            config.sep_norm_token_id,
        )

        f0_bin = config.default_f0_bin
        prompt_ids = _build_prosody_prompt_ids(tokenizer, normalized_text, f0_bin)

        num_words = len(normalized_text.split())
        annot_block = _build_nar_annotation_block(config, num_words)
        prompt_ids = list(prompt_ids) + annot_block

        logger.debug(
            "process_text_norm_to_prosody: req=%s, num_words=%d, prompt_len=%d",
            getattr(source_output, "request_id", "?"),
            num_words,
            len(prompt_ids),
        )

        results.append(OmniTokensPrompt(prompt_token_ids=prompt_ids))
    return results


def process_prosody_to_tts(
    source_outputs: list,
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list:
    """Orchestrator-path processor: Stage 1 (prosody) → Stage 2 (StyleTTS2).

    Extracts prosody tokens from source multimodal output, builds TTS
    payload (phonemes, prsinf, boundaries, speaker embedding), and packs
    into prompt_token_ids for the StyleTTS2 decoder.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    results = []
    for source_output in source_outputs:
        mm_output = _extract_multimodal_output(source_output)

        prosody_tensor = mm_output.get("prosody_tokens") if mm_output else None
        if prosody_tensor is None:
            logger.warning(
                "process_prosody_to_tts: no prosody_tokens for req=%s, mm_keys=%s",
                getattr(source_output, "request_id", "?"),
                list(mm_output.keys()) if mm_output else "None",
            )
            continue

        if isinstance(prosody_tensor, torch.Tensor):
            annot_tokens = prosody_tensor.tolist()
        elif isinstance(prosody_tensor, list):
            annot_tokens = prosody_tensor
        else:
            annot_tokens = list(prosody_tensor)

        if not annot_tokens:
            logger.warning(
                "process_prosody_to_tts: empty prosody_tokens for req=%s",
                getattr(source_output, "request_id", "?"),
            )
            continue

        config, tokenizer = _get_orchestrator_config_and_tokenizer()

        prompt_ids = list(getattr(source_output, "prompt_token_ids", None) or [])
        normalized_text = _extract_normalized_text_from_prompt(
            tokenizer,
            prompt_ids,
            config.sep_norm_token_id,
            config.sep_f0_token_id,
        )
        if not normalized_text:
            logger.warning(
                "process_prosody_to_tts: no normalized_text for req=%s",
                getattr(source_output, "request_id", "?"),
            )
            continue

        g_pre, _ = compute_preamble_layout(
            config.emotion_control,
            list(config.nar_global_dims),
            config.compact_preamble,
        )
        output_nums = _nar_block_to_nums(annot_tokens, config.num_start_id, g_pre)
        prsinf = _output_nums_to_prsinf(output_nums)

        tts_text = re.sub(r"^\[SPEAKER\s+\S+\]\s*", "", normalized_text)
        tts_text = re.sub(r"^\[SPEAKER\s+\S+\s+", "", tts_text)
        tts_text = tts_text.replace("_", " ")
        ps = _text_to_phonemes(tts_text)
        tokens = _text_cleaner(ps)
        tokens.insert(0, 0)
        is_sil = [1 if _ID2SYM.get(tk, "") in _PUNCTUATION + _PAD else 0 for tk in tokens]
        if is_sil and is_sil[-1] == 0:
            tokens.append(0)
            is_sil.append(1)
        indices_word = _find_consecutive_indices(is_sil)
        boundaries = np.array(indices_word)[:, 1:].T

        model_dir = os.environ.get("GRANITE_PROSODY_LM_MODEL_PATH", "")
        tts_dir = os.path.join(model_dir, "stage2_tts") if model_dir else ""
        spk_emb = _load_speaker_embedding(tts_dir)

        packed = pack_tts_payload(
            phoneme_tokens=tokens,
            prsinf=prsinf,
            boundaries=boundaries,
            speaker_embedding=spk_emb,
        )

        results.append(OmniTokensPrompt(prompt_token_ids=packed.tolist()))
    return results


# ─── Stage 1 → Stage 2: prosody → TTS ─────────────────────────────────────────

# Phoneme symbol table (matches StyleTTS2's tts_local/utils.py)
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
_SYM2ID: dict[str, int] = {s: i for i, s in enumerate(_SYMBOLS)}
_ID2SYM: dict[int, str] = {i: s for s, i in _SYM2ID.items()}


def pack_tts_payload(
    phoneme_tokens: list[int],
    prsinf: np.ndarray,
    boundaries: np.ndarray,
    speaker_embedding: torch.Tensor,
) -> torch.Tensor:
    """Pack TTS data into a single int64 tensor for inter-stage transport.

    Layout: [header(8)] + phoneme_tokens + prsinf_flat + boundaries_flat + spk_emb_bytes
    Header: n_phonemes, prsinf_r, prsinf_c, bound_r, bound_c, spk_r, spk_c, spk_nbytes
    Speaker embedding is bit-cast from float32 to int32 pairs.
    """
    spk = speaker_embedding.detach().cpu().float().flatten()
    spk_bytes = spk.view(torch.int32)

    header = torch.tensor(
        [
            len(phoneme_tokens),
            prsinf.shape[0],
            prsinf.shape[1],
            boundaries.shape[0],
            boundaries.shape[1],
            spk.shape[0],
            0,
            spk_bytes.shape[0],
        ],
        dtype=torch.int64,
    )

    parts = [
        header,
        torch.tensor(phoneme_tokens, dtype=torch.int64),
        torch.from_numpy(prsinf).long().flatten(),
        torch.from_numpy(boundaries).long().flatten(),
        spk_bytes.to(torch.int64),
    ]
    return torch.cat(parts)


def unpack_tts_payload(packed: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack a tensor produced by pack_tts_payload back to TTS data dict."""
    h = packed[:8]
    n_ph, pr_r, pr_c, bd_r, bd_c, spk_n, _, spk_nb = h.tolist()
    n_ph, pr_r, pr_c, bd_r, bd_c, spk_n, spk_nb = (
        int(n_ph),
        int(pr_r),
        int(pr_c),
        int(bd_r),
        int(bd_c),
        int(spk_n),
        int(spk_nb),
    )

    offset = 8
    phoneme_tokens = packed[offset : offset + n_ph].unsqueeze(0)
    offset += n_ph

    prsinf = packed[offset : offset + pr_r * pr_c].reshape(pr_r, pr_c)
    offset += pr_r * pr_c

    boundaries = packed[offset : offset + bd_r * bd_c].reshape(bd_r, bd_c)
    offset += bd_r * bd_c

    spk_int = packed[offset : offset + spk_nb].to(torch.int32)
    spk_emb = spk_int.view(torch.float32).reshape(1, spk_n)

    return {
        "phoneme_tokens": phoneme_tokens,
        "prsinf": prsinf,
        "boundaries": boundaries,
        "speaker_embedding": spk_emb,
    }


def _text_cleaner(text: str) -> list[int]:
    """Convert IPA phoneme string to StyleTTS2 token IDs."""
    return [_SYM2ID[c] for c in text if c in _SYM2ID]


def _find_consecutive_indices(arr: list[int]) -> list[tuple[int, int, int]]:
    """Group consecutive equal values. Returns [(value, start, end), ...]."""
    result = []
    start = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            result.append((arr[start], start, i))
            start = i
    result.append((arr[start], start, len(arr)))
    return result


def _nar_block_to_nums(
    decoded: list[int],
    num_start_id: int,
    g_pre: int = 6,
) -> list[list[int]]:
    """Convert NAR annotation block token IDs to nested-list format for TTS.

    Input: [<SIL>] + [preamble(g_pre)] + [word_col(6)]*N + [SEP_2]
    Returns [[sil0], [dur1,p1,p2,p3,p4], [sil1], ...,[silN]]
    """
    block = decoded[1:-1]  # strip <SIL> prefix and [SEP_2] suffix
    n_words = (len(block) - g_pre) // 6
    result: list[list[int]] = []
    result.append([block[g_pre - 1] - num_start_id])
    for w in range(n_words):
        base = g_pre + w * 6
        result.append([block[base + i] - num_start_id for i in range(5)])
        result.append([block[base + 5] - num_start_id])
    return result


def _output_nums_to_prsinf(output_nums: list[list[int]]) -> np.ndarray:
    """Convert nested-list prosody output to prsinf array (n_words+1, 5).

    Silence entries ([sil_dur]) → [sil_dur, 512, 512, 512, 512]
    Word entries ([dur, p1, p2, p3, p4]) → values with 512→256 mapping
    """
    extracted = []
    for prs_vec in output_nums:
        if len(prs_vec) == 1:
            new_prs = [512, 512, 512, 512, 512]
            new_prs[0] = prs_vec[0]
        else:
            new_prs = [256 if v == 512 else v for v in prs_vec]
        extracted.append(new_prs)
    return np.array(extracted)


@lru_cache(maxsize=1)
def _load_phonemizer_backend(language: str = "en-us"):
    """Lazily load the phonemizer espeak backend."""
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(language, preserve_punctuation=True, with_stress=True)


def _text_to_phonemes(text: str, language: str = "en-us") -> str:
    """Convert text to IPA phonemes using espeak backend."""
    backend = _load_phonemizer_backend(language)
    outputs = backend.phonemize([text])
    output = outputs[0] if outputs else ""
    if text[:1] == " " and output[:1] != " ":
        output = " " + output
    if text[:1] != " " and output[:1] == " ":
        output = output[1:]
    if text[-1:] == " " and output[-1:] != " ":
        output = output + " "
    if text[-1:] != " " and output[-1:] == " ":
        output = output[:-1]
    j = 0
    while j < len(output) - 1:
        if output[j] == " " and output[j + 1] in _PUNCTUATION:
            output = output[:j] + output[j + 1 :]
        j += 1
    return output


@lru_cache(maxsize=1)
def _load_speaker_embedding(model_dir: str) -> torch.Tensor:
    """Load precomputed speaker style embedding from the TTS model directory."""
    pt_path = os.path.join(model_dir, "speaker_embedding.pt")
    if os.path.isfile(pt_path):
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        logger.warning(
            "No speaker_embedding.pt found in %s; using zeros",
            model_dir,
        )
        return torch.zeros(1, 128, dtype=torch.float32)
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    return emb


def _extract_normalized_text_from_prompt(
    tokenizer,
    prompt_token_ids: list[int],
    sep_norm_id: int,
    sep_f0_id: int,
) -> str:
    """Extract the normalized text from the prosody stage's prompt tokens.

    The prompt format (Granite chat template) is:
      <|start_of_role|>system<|end_of_role|>...<|end_of_text|>
      <|start_of_role|>assistant<|end_of_role|>norm_text[SEP_NORM]...
    We decode the portion between the last <|end_of_role|> and [SEP_NORM].
    """
    sep_norm_pos = None
    for i, tid in enumerate(prompt_token_ids):
        if tid == sep_norm_id:
            sep_norm_pos = i
            break
    if sep_norm_pos is None:
        logger.warning(
            "_extract_normalized_text_from_prompt: sep_norm_id=%d NOT FOUND "
            "in prompt_token_ids (len=%d). Decoded first 50 tokens: %r",
            sep_norm_id,
            len(prompt_token_ids),
            tokenizer.decode(prompt_token_ids[:50], skip_special_tokens=False),
        )
        return ""
    end_of_role_id = tokenizer.encode(
        "<|end_of_role|>",
        add_special_tokens=False,
    )
    if not end_of_role_id:
        logger.warning("_extract_normalized_text_from_prompt: <|end_of_role|> encode returned empty")
        return ""
    eor_id = end_of_role_id[0]
    start_pos = 0
    for i in range(sep_norm_pos - 1, -1, -1):
        if prompt_token_ids[i] == eor_id:
            start_pos = i + 1
            break
    norm_ids = prompt_token_ids[start_pos:sep_norm_pos]
    result = tokenizer.decode(norm_ids, skip_special_tokens=False)
    return result
