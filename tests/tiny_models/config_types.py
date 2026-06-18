"""
Common definitions for controlling what tests run where.
"""

from collections.abc import Callable
from enum import StrEnum, auto
from typing import NamedTuple, TypeAlias

from pytest import MarkDecorator

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni

# All builder funcs take no params and return a path
TinyDiffusionBuilder: TypeAlias = Callable[[], str]


class DiffAccs(StrEnum):
    """Supported acceleration types / test settings for Diffusion Models."""

    HSDP = auto()
    TEA_CACHE = auto()
    CACHE_DIT = auto()
    SEQUENCE_PARALLEL = auto()
    CFG_PARALLEL = auto()
    TENSOR_PARALLEL = auto()
    CPU_OFFLOAD = auto()
    VAE_PATCH_PARALLEL = auto()


class DiffTasks(StrEnum):
    """Supported tasks for Diffusion Models."""

    TEXT_TO_IMAGE = auto()
    IMAGE_EDIT = auto()
    # Text to video, text to audio, etc should be added here as needed


class DiffusionModelTestOpts(NamedTuple):
    """Configuration for one Diffusion model's tests."""

    builder: TinyDiffusionBuilder

    # Actual tasks which controls the tests actually run
    supported_tasks: list[DiffTasks]

    # Accelerations to be run together for this model; we currently specify
    # this explicitly because the time to start a model is nontrivial, even
    # for tiny models.
    test_groups: list[None | list[DiffAccs]]

    # Pytest Marks for this model. This may be useful for selecting which models
    # we want to run where, similar to the way vLLM's multimodal tests mark some
    # as core models to always run in the CI.
    # Example: https://github.com/vllm-project/vllm/blob/v0.23.0/tests/models/multimodal/generation/test_common.py#L131
    marks: list[MarkDecorator] | None = None


### Mappings & utils for building offline Omni() instances given a list of enabled accelerations
ACC_OMNI_KWARGS = {
    DiffAccs.VAE_PATCH_PARALLEL: {"vae_use_tiling": True},
    DiffAccs.CPU_OFFLOAD: {"enable_cpu_offload": True},
    DiffAccs.CACHE_DIT: {"cache_backend": "cache_dit"},
    DiffAccs.TEA_CACHE: {"cache_backend": "tea_cache"},
}

ACC_PARALLEL_KWARGS = {
    DiffAccs.HSDP: {"use_hsdp": True, "hsdp_shard_size": 2},
    DiffAccs.TENSOR_PARALLEL: {"tensor_parallel_size": 2},
    DiffAccs.CFG_PARALLEL: {"cfg_parallel_size": 2},
    DiffAccs.VAE_PATCH_PARALLEL: {"vae_patch_parallel_size": 2},
    # For SP, we don't run ring here to conserve devices, since it is easy
    # to blow up the number of needed devices fast. Compatibility for ring
    # and ulysses together should be tested generically outside of these tests.
    DiffAccs.SEQUENCE_PARALLEL: {"ulysses_degree": 2},
}

# TODO - combine with that ^, this is messy
ACC_DEVICE_COUNT_KEYS = {
    DiffAccs.TENSOR_PARALLEL: "tensor_parallel_size",
    DiffAccs.CFG_PARALLEL: "cfg_parallel_size",
    DiffAccs.SEQUENCE_PARALLEL: "ulysses_degree",
    DiffAccs.VAE_PATCH_PARALLEL: "vae_patch_parallel_size",
    DiffAccs.HSDP: "hsdp_shard_size",
}


def get_required_device_count(accelerations: list[DiffAccs] | None) -> int:
    """Compute the minimum number of devices needed for a set of accelerations.
    The total is the product of all parallel dimensions (defaulting to 1).

    If not enough devices are available for a test group's accelerations,
    that test will be skipped."""
    count = 1
    if accelerations is None:
        return count

    for acc in accelerations:
        key = ACC_DEVICE_COUNT_KEYS.get(acc)
        if key is not None:
            count *= ACC_PARALLEL_KWARGS[acc][key]
    return count


def build_parallel_config_from_diff_accelerations(accelerations: list[DiffAccs]) -> DiffusionParallelConfig | None:
    """Given a list of accelerations pertaining to the current test group,
    build the parallel config needed for the Omni() object (if any)."""
    config_kwargs = {}
    for acc in accelerations:
        update_dict = ACC_PARALLEL_KWARGS.get(acc, {})
        config_kwargs.update(update_dict)
    if config_kwargs:
        return DiffusionParallelConfig(**config_kwargs)
    return None


def build_omni_from_diff_accelerations(accelerations: list[DiffAccs] | None, **kwargs) -> Omni:
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
        update_dict = ACC_OMNI_KWARGS.get(acc, {})
        acc_kwargs.update(update_dict)

    # Keys passed through should mostly be things like enforce_eager;
    # if there's overlap, it's probably due to a misconfiguration
    shared_keys = acc_kwargs.keys() & kwargs.keys()
    if shared_keys:
        raise ValueError(f"Explicit Omni kwargs and inferred Omni kwargs for accelerations overlap: {shared_keys}")
    omni_kwargs = {**acc_kwargs, **kwargs}
    return Omni(**omni_kwargs)
