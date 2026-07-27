import sys
from pathlib import Path

sys.path.append("/home/alex-jw-brooks/vllm-omni")
from transformers import AutoConfig, AutoModelForMultimodalLM

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


# def tiny_qwen3_omni_builder() -> str:
#     """Build a tiny (3 stage) Qwen3Omni model from vendored configs."""
def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""


if __name__ == "__main__":
    TINY_QWEN_DIR = TINY_CONFIGS_DIR / "qwen3_omni"

    config = AutoConfig.from_pretrained(TINY_QWEN_DIR)
    model = AutoModelForMultimodalLM.from_config(config)
    print("OK")
