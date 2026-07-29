import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForMultimodalLM
from vllm.config.vllm import set_current_vllm_config
from vllm.transformers_utils.processors.bagel import BagelProcessor

from tests.model_tests.utils import (
    TINY_CONFIGS_DIR,
    build_tiny_from_configs,
    copy_configs_to_model_dir,
    get_tiny_model_path,
    get_vllm_config,
    stub_vllm_parallel_state,
    unfuse_packed_state_dict,
)
from vllm_omni.diffusion.models.bagel.autoencoder import AutoEncoder
from vllm_omni.diffusion.models.bagel.pipeline_bagel import default_ae_params
from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelForConditionalGeneration,
)


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
def build_bagel_ema_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """Create the EMA tensors for a Bagel tiny model."""
    # NOTE: model_dir here is the tiny model dir with the small config
    # that we want to generate the randomly initialized EMA tensors for
    vllm_config = get_vllm_config(model_dir, trust_remote_code=True)
    with set_current_vllm_config(vllm_config):
        model = OmniBagelForConditionalGeneration(vllm_config=vllm_config, prefix="")

    sd = model.state_dict()
    # vit_pos_embed.pos_embed is a buffer needed by DiT loader
    sd["vit_pos_embed.pos_embed"] = model.vit_pos_embed.pos_embed

    ckpt = unfuse_packed_state_dict(sd, OmniBagelForConditionalGeneration.packed_modules_mapping)

    # This is a DiT-only module not present in the OmniBagel for cond gen class
    # so for now we calculate the values directly based on the config
    hf_cfg = vllm_config.model_config.hf_config
    H = hf_cfg.llm_config.hidden_size
    lpd = hf_cfg.latent_patch_size**2 * hf_cfg.vae_config["z_channels"]
    ckpt["llm2vae.weight"] = torch.zeros(lpd, H)
    ckpt["llm2vae.bias"] = torch.zeros(lpd)

    # Ensure that the checkpoint is valid by making sure it's reloadable
    model.load_weights(ckpt.items())
    return ckpt


def build_tiny_bagel() -> str:
    """Build a tiny Bagel model."""
    model_name = "bagel"
    ref_model = "ByteDance-Seed/BAGEL-7B-MoT"
    model_dir = get_tiny_model_path(model_name)
    model_dir_path = Path(model_dir)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    cfg_dir = TINY_CONFIGS_DIR / model_name

    with open(cfg_dir / "config.json") as cfg_ptr:
        cfg = json.load(cfg_ptr)

    # Ensure each subconfig exists and copy it into the model dir
    copy_configs_to_model_dir(
        cfg_dir,
        model_dir_path,
        ["config.json", "llm_config.json", "vit_config.json"],
    )

    # Save the Bagel processor (uses a smaller image size)
    image_size = cfg["vit_config"]["image_size"]
    proc = BagelProcessor.from_pretrained(ref_model)
    proc.image_processor.size = {"height": image_size, "width": image_size}
    proc.save_pretrained(model_dir)

    # Save the Autoencoder tensors
    ae = AutoEncoder(default_ae_params())
    save_file(ae.state_dict(), model_dir_path / "ae.safetensors")

    # Save the ema tensors; the easiest way to create this is
    # through the vLLM module, so we need to stub the parallel
    # state and create a vllm config context.
    stub_vllm_parallel_state()
    ema_state_dict = build_bagel_ema_state_dict(model_dir)
    save_file(ema_state_dict, model_dir_path / "ema.safetensors")
    torch.set_default_dtype(prev_dtype)
    return model_dir


def _build_tiny_qwen3_omni(*, enable_audio_output: bool) -> str:
    """Build a Qwen3Omni model (thinker only or multistage)."""
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    suffix = "qwen3_omni" if enable_audio_output else "qwen3_omni_thinker"
    model_dir = get_tiny_model_path(suffix)
    config = AutoConfig.from_pretrained(TINY_CONFIGS_DIR / "qwen3_omni")
    # If enable_audio_output is True, we run thinker only
    config.enable_audio_output = enable_audio_output
    model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    model.save_pretrained(model_dir)
    snapshot_download(
        model_id,
        allow_patterns=["tokenizer*", "merges.txt", "vocab.json", "preprocessor_config.json"],
        local_dir=model_dir,
    )
    return model_dir


def tiny_qwen3_omni_builder() -> str:
    """Build a tiny Qwen3Omni model with all 3 stages."""
    return _build_tiny_qwen3_omni(enable_audio_output=True)


def tiny_qwen3_omni_thinker_builder() -> str:
    """Build a tiny thinker-only Qwen3Omni model."""
    return _build_tiny_qwen3_omni(enable_audio_output=False)
