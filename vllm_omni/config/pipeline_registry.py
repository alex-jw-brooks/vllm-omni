# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central declarative registry of all vllm-omni pipelines.

Mirrors the pattern in ``vllm/model_executor/models/registry.py``: each entry
is ``model_type -> (module_path, variable_name)``, and the module is imported
lazily on first lookup (see ``_LazyPipelineRegistry`` in
``vllm_omni/config/stage_config.py``). Keeping every pipeline declared in one
file makes it easy to spot a missing registration, which was the original
motivation in https://github.com/vllm-project/vllm-omni/issues/2887 (item 4).

Per-model ``pipeline.py`` modules still define the ``PipelineConfig`` instance;
they just no longer need to self-register via ``register_pipeline(...)``.

Adding a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. Add one line to ``OMNI_PIPELINES`` below.

Plain single-stage diffusion models continue to use the
``_create_default_diffusion_stage_cfg`` fallback in ``async_omni_engine.py``.
The empty ``_DIFFUSION_PIPELINES`` placeholder previously here (#2915) was
removed once #2987 (which would have populated it) was deferred.

``register_pipeline(config)`` in ``stage_config`` is still supported for
out-of-tree plugins and tests that create pipelines at runtime; those override
the entries declared here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from transformers import PreTrainedConfig

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_PIPELINE
from vllm_omni.model_executor.models.bagel.pipeline import (
    BAGEL_PIPELINE,
    BAGEL_SINGLE_STAGE_PIPELINE,
    BAGEL_THINK_PIPELINE,
)
from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_PIPELINE
from vllm_omni.model_executor.models.covo_audio.pipeline import COVO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.dreamzero.pipeline import DREAMZERO_PIPELINE
from vllm_omni.model_executor.models.dynin_omni.pipeline import DYNIN_OMNI_PIPELINE
from vllm_omni.model_executor.models.fish_speech.pipeline import FISH_SPEECH_PIPELINE
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE
from vllm_omni.model_executor.models.glm_tts.pipeline import GLM_TTS_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import HIGGS_AUDIO_V2_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v3.pipeline import HIGGS_AUDIO_V3_PIPELINE
from vllm_omni.model_executor.models.hunyuan_image3.pipeline import (
    HUNYUAN_IMAGE3_AR_PIPELINE,
    HUNYUAN_IMAGE3_DIT_PIPELINE,
    HUNYUAN_IMAGE3_PIPELINE,
)
from vllm_omni.model_executor.models.indextts2.pipeline import INDEXTTS2_PIPELINE
from vllm_omni.model_executor.models.lance.pipeline import LANCE_PIPELINE
from vllm_omni.model_executor.models.mimo_audio.pipeline import MIMO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.ming_flash_omni.pipeline import (
    MING_FLASH_OMNI_IMAGE_PIPELINE,
    MING_FLASH_OMNI_PIPELINE,
    MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    MING_FLASH_OMNI_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.ming_tts.pipeline import MING_TTS_PIPELINE
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.models.moss_tts.pipeline import MOSS_TTS_PIPELINE, MOSS_TTS_REALTIME_PIPELINE
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import MOSS_TTS_NANO_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    QWEN2_5_OMNI_PIPELINE,
    QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import resolve_qwen3_omni_pipeline
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.voxcpm2.pipeline import VOXCPM2_PIPELINE
from vllm_omni.model_executor.models.voxtral_tts.pipeline import VOXTRAL_TTS_PIPELINE

PipelineResolverFunc: TypeAlias = Callable[[PreTrainedConfig | None], PipelineConfig]

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc] = {
    "aura_omni": AURA_OMNI_PIPELINE,
    "qwen2_5_omni": QWEN2_5_OMNI_PIPELINE,
    "qwen2_5_omni_thinker_only": QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
    "qwen3_omni_moe": resolve_qwen3_omni_pipeline,
    "qwen3_tts": QWEN3_TTS_PIPELINE,
    "covo_audio": COVO_AUDIO_PIPELINE,
    "bagel": BAGEL_PIPELINE,
    "bagel_think": BAGEL_THINK_PIPELINE,
    "bagel_single_stage": BAGEL_SINGLE_STAGE_PIPELINE,
    "lance": LANCE_PIPELINE,
    "dreamzero": DREAMZERO_PIPELINE,
    "glm_image": GLM_IMAGE_PIPELINE,
    "hunyuan_image_3_moe": HUNYUAN_IMAGE3_PIPELINE,
    "hunyuan_image3_ar": HUNYUAN_IMAGE3_AR_PIPELINE,
    "hunyuan_image3_dit": HUNYUAN_IMAGE3_DIT_PIPELINE,
    "voxcpm2": VOXCPM2_PIPELINE,
    "cosyvoice3": COSYVOICE3_PIPELINE,
    "mimo_audio": MIMO_AUDIO_PIPELINE,
    "ming_tts": MING_TTS_PIPELINE,
    "voxtral_tts": VOXTRAL_TTS_PIPELINE,
    "glm_tts": GLM_TTS_PIPELINE,
    "fish_qwen3_omni": FISH_SPEECH_PIPELINE,
    "ming_flash_omni": MING_FLASH_OMNI_PIPELINE,
    "ming_flash_omni_tts": MING_FLASH_OMNI_TTS_PIPELINE,
    "ming_flash_omni_thinker_only": MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    "ming_flash_omni_image": MING_FLASH_OMNI_IMAGE_PIPELINE,
    "moss_tts_nano": MOSS_TTS_NANO_PIPELINE,
    "moss_tts_delay": MOSS_TTS_PIPELINE,
    "moss_tts_realtime": MOSS_TTS_REALTIME_PIPELINE,
    "minicpmo_4_5": MINICPMO_4_5_PIPELINE,
    "higgs_audio_v2": HIGGS_AUDIO_V2_PIPELINE,
    "higgs_multimodal_qwen3": HIGGS_AUDIO_V3_PIPELINE,
    "dynin_omni": DYNIN_OMNI_PIPELINE,
    "indextts2": INDEXTTS2_PIPELINE,
}
