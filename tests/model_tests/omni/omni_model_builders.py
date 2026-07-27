import os
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"
TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


def tiny_qwen3_omni_builder() -> str:
    """Build a tiny Qwen3Omni model (all 3 stages) & return saved path."""
    config = AutoConfig.from_pretrained(TINY_CONFIGS_DIR / "qwen3_omni")
    model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    outdir = os.path.join(TINY_MODEL_DIR, "qwen3_omni")
    model.save_pretrained(outdir)
    return outdir
