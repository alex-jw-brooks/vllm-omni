import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForMultimodalLM

from tests.helpers.tiny_model import build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"
TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


### Single stage diffusion models
def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")


### Omni models / multi-stage Diffusion Models
def _build_tiny_qwen3_omni(*, enable_audio_output: bool) -> str:
    """Build a Qwen3Omni model (thinker only or multistage)."""
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    suffix = "qwen3_omni" if enable_audio_output else "qwen3_omni_thinker"
    config = AutoConfig.from_pretrained(TINY_CONFIGS_DIR / "qwen3_omni")
    # If enable_audio_output is True, we run thinker only
    config.enable_audio_output = enable_audio_output
    model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    outdir = os.path.join(TINY_MODEL_DIR, suffix)
    model.save_pretrained(outdir)
    snapshot_download(
        model_id,
        allow_patterns=["tokenizer*", "merges.txt", "vocab.json", "preprocessor_config.json"],
        local_dir=outdir,
    )
    return outdir


def tiny_qwen3_omni_builder() -> str:
    """Build a tiny Qwen3Omni model with all 3 stages."""
    return _build_tiny_qwen3_omni(enable_audio_output=True)


def tiny_qwen3_omni_thinker_builder() -> str:
    """Build a tiny thinker-only Qwen3Omni model."""
    return _build_tiny_qwen3_omni(enable_audio_output=False)
