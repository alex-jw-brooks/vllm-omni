"""
Utilities for resolving real models to their tiny model equivalents.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import vllm.distributed.parallel_state as ps
from diffusers import ModelMixin
from diffusers.pipelines.pipeline_loading_utils import _get_pipeline_class, simple_get_class_obj
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import AutoConfig, PreTrainedModel
from vllm.config import ModelConfig, VllmConfig

from tests.model_tests.config_types import ACC_DESCRIPTORS, DiffusionAccs
from vllm_omni.diffusion.data import DiffusionParallelConfig, resolve_model_class_name
from vllm_omni.entrypoints.omni import Omni

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")
TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


logger = logging.getLogger(__name__)


def resolve_tiny_model_path(model: str) -> str:
    """Given a real model name/path, resolve it to a tiny model path.

    Raises ValueError if the pipeline class cannot be determined (invalid
    model). Returns the original model path if no tiny builder exists yet."""
    # NOTE: For now, this is to avoid a circular dependency, since this is
    # the only place we actually depend on the test settings.
    from tests.model_tests.model_settings import MODEL_SETTINGS

    pipeline_class = resolve_model_class_name(model)
    if pipeline_class is None:
        raise ValueError(
            f"Cannot resolve pipeline class for model: {model}. The model path may be invalid or its config unreadable."
        )

    test_opts = MODEL_SETTINGS.get(pipeline_class)
    if test_opts is None:
        logger.warning(
            "No tiny model builder for pipeline %s (model: %s). Using original model.",
            pipeline_class,
            model,
        )
        return model

    return test_opts.builder()


### helpers for building Diffusion Models
def get_required_device_count(accelerations: list[DiffusionAccs] | None) -> int:
    """Compute the minimum number of devices needed for a set of accelerations.
    The total is the product of all parallel dimensions (defaulting to 1).

    If not enough devices are available for a test group's accelerations,
    that test will be skipped."""
    count = 1
    if accelerations is None:
        return count

    for acc in accelerations:
        descriptor = ACC_DESCRIPTORS[acc]
        if descriptor.device_count_key is not None:
            count *= descriptor.omni_parallel_kwargs[descriptor.device_count_key]
    return count


def build_parallel_config_from_diff_accelerations(accelerations: list[DiffusionAccs]) -> DiffusionParallelConfig | None:
    """Given a list of accelerations pertaining to the current test group,
    build the parallel config needed for the Omni() object (if any)."""
    config_kwargs = {}
    for acc in accelerations:
        update_dict = ACC_DESCRIPTORS[acc].omni_parallel_kwargs
        config_kwargs.update(update_dict)
    if config_kwargs:
        return DiffusionParallelConfig(**config_kwargs)
    return None


### Offline Omni() object builder
def build_omni_from_diff_accelerations(accelerations: list[DiffusionAccs] | None, **kwargs) -> Omni:
    """Given one or more acceleration types, build the corresponding Omni() object."""
    # Coerce to a list and build the parallel config, since that depends on the accelerations
    if accelerations is None:
        accelerations = []
    parallel_config = build_parallel_config_from_diff_accelerations(accelerations)

    # Then add anything else that's a top-level kwarg
    acc_kwargs = {}
    if parallel_config is not None:
        acc_kwargs["parallel_config"] = parallel_config
    for acc in accelerations:
        update_dict = ACC_DESCRIPTORS[acc].omni_kwargs
        acc_kwargs.update(update_dict)

    # Keys passed through should mostly be things like enforce_eager;
    # if there's overlap, it's probably due to a misconfiguration
    shared_keys = acc_kwargs.keys() & kwargs.keys()
    if shared_keys:
        raise ValueError(f"Explicit Omni kwargs and inferred Omni kwargs for accelerations overlap: {shared_keys}")
    omni_kwargs = {**acc_kwargs, **kwargs}
    return Omni(**omni_kwargs)


### Online server flag builder
def build_server_args_from_diff_accelerations(accelerations: list[DiffusionAccs] | None) -> list[str]:
    """Given one or more acceleration types, build the corresponding CLI args
    for launching an OmniServer subprocess."""
    if accelerations is None:
        return []
    args = []
    for acc in accelerations:
        acc_cli_args = ACC_DESCRIPTORS[acc].cli_args
        args.extend(acc_cli_args)
    return args


### Reusable builder utils for making tiny models from HF libs
def get_tiny_model_path(name: str) -> str:
    """Get the path to this model's tiny dir to export to."""
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def _diffusers_from_config(cls, pretrained_model_name_or_path, **kwargs):
    """Replacement for ModelMixin.from_pretrained that initializes random weights."""
    subfolder = kwargs.get("subfolder")
    load_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = cls.load_config(pretrained_model_name_or_path, **load_kwargs)
    return cls.from_config(config)


def _transformers_from_config(cls, pretrained_model_name_or_path, **kwargs):
    """Replacement for PreTrainedModel.from_pretrained that initializes random weights."""
    subfolder = kwargs.get("subfolder")
    load_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    return cls(config)


def build_tiny_from_configs(pipeline_name: str, model_id: str, configs_dir: str | Path) -> str:
    """Build a tiny diffusion model from vendored configs with random weights.

    Mirrors the component loading loop in DiffusionPipeline.from_pretrained,
    but monkeypatches from_pretrained on ModelMixin and PreTrainedModel to
    initialize with random weights instead of loading checkpoint files.
    Components with vendored configs use those; others load configs from
    the upstream HF model.

    Args:
        pipeline_name: Name of the pipeline (used as output directory name).
        model_id: HuggingFace model ID for loading upstream components
            (tokenizer, scheduler) that don't have vendored configs.
        configs_dir: Path to the directory containing vendored config files
            (model_index.json and per-component config.json).

    Returns:
        Path to the saved tiny model directory with safetensors weights.
    """
    model_dir = get_tiny_model_path(pipeline_name)
    config_dir = Path(configs_dir)

    config_dict = DiffusionPipeline.load_config(config_dir)
    pipeline_cls = _get_pipeline_class(DiffusionPipeline, config=config_dict)

    init_dict, _, _ = pipeline_cls.extract_init_dict(config_dict)

    # Pop non-component entries (optional pipeline kwargs like is_distilled),
    # same as DiffusionPipeline.from_pretrained lines 345-350
    _, optional_kwargs = DiffusionPipeline._get_signature_keys(pipeline_cls)
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}

    with (
        patch.object(ModelMixin, "from_pretrained", classmethod(_diffusers_from_config)),
        patch.object(PreTrainedModel, "from_pretrained", classmethod(_transformers_from_config)),
    ):
        for name, (library_name, class_name) in init_dict.items():
            cls = simple_get_class_obj(library_name, class_name)

            # Use vendored config dir if available, otherwise upstream model
            if (config_dir / name).exists():
                init_kwargs[name] = cls.from_pretrained(config_dir / name)
            else:
                init_kwargs[name] = cls.from_pretrained(model_id, subfolder=name)

    pipe = pipeline_cls(**init_kwargs)
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir


### Misc helpers for building more complex models that depend on external implementations
def copy_configs_to_model_dir(cfg_dir: Path, model_dir: Path, config_names: list[str]):
    """Ensure configs exist in the tiny config dir, then copy them to the model output dir
    with tiny weights.."""

    copy_paths = [cfg_dir / fname for fname in os.listdir(cfg_dir) if fname in config_names]
    assert len(copy_paths) == len(config_names)
    for cfg_name in config_names:
        cfg_path = cfg_dir / cfg_name
        out_path = model_dir / cfg_name
        assert os.path.isfile(cfg_path)
        shutil.copy(cfg_path, out_path)


def get_vllm_config(model_path: str, trust_remote_code: bool):
    """Given a model path, create a vLLM config for it."""
    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=trust_remote_code,
    )
    return VllmConfig(model_config=model_config)


def stub_vllm_parallel_state():
    """Create a mock parallel state for vLLM; this is needed in some
    tricky cases where models run through their own external libraries,
    and the easiest way to create a random tiny model is to initialize
    the vLLM component, which may contain modules requiring an initialized
    parallel state + vLLM config (e.g., Bagel)."""
    fake = MagicMock(
        rank_in_group=0,
        world_size=1,
        is_first_rank=True,
        is_last_rank=True,
    )
    ps._TP = ps._PP = fake


def unfuse_packed_state_dict(
    sd: dict[str, torch.Tensor],
    packed_mapping: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    """Reverse packed_modules_mapping for vLLM modules back to their checkpoint format.

    NOTE: This is currently written assuming that the tiny model is written to use
    MHA, and not GQA.
    """
    out: dict[str, torch.Tensor] = {}
    seen_fused: set[str] = set()
    # longest first so "mlp_moe_gen.gate_up_proj" matches before "gate_up_proj"
    ordered = sorted(packed_mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for key, val in sd.items():
        for fused, sources in ordered:
            sep = f".{fused}."
            if sep not in key:
                continue
            seen_fused.add(fused)
            prefix, _, param_type = key.rpartition(sep)
            for src, chunk in zip(sources, torch.chunk(val, len(sources), dim=0)):
                out[f"{prefix}.{src}.{param_type}"] = chunk
            break
        else:
            out[key] = val

    missing = set(packed_mapping) - seen_fused
    if missing:
        raise ValueError(f"packed_mapping entries had no matching keys: {missing}")
    return out
