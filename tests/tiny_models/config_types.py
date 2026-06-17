"""
Common definitions for controlling what tests run where.
"""

from collections.abc import Callable
from enum import StrEnum, auto
from typing import NamedTuple, TypeAlias

from pytest import MarkDecorator

from tests.tiny_models import diff_acc_builders as dac_build
from vllm_omni.entrypoints.omni import Omni

# All builder funcs take no params and return a path
TinyDiffusionBuilder: TypeAlias = Callable[[], str]


class DiffAccs(StrEnum):
    HSDP = auto()
    TEA_CACHE = auto()
    CACHE_DIT = auto()
    SEQUENCE_PARALLEL = auto()
    CFG_PARALLEL = auto()
    TENSOR_PARALLEL = auto()
    CPU_OFFLOAD = auto()
    VAE_PATCH_PARALLEL = auto()


class DiffTasks(StrEnum):
    TEXT_TO_IMAGE = auto()
    IMAGE_EDIT = auto()
    # Text to video, text to audio, etc should be added here as needed


class DiffusionModelTestOpts(NamedTuple):
    builder: TinyDiffusionBuilder

    # Actual tasks which controls the tests actually run
    supported_tasks: list[DiffTasks] | None = None

    # Accelerations supported by this model
    supported_accelerations: list[DiffAccs] | None = None

    # Extra args to passed to sampling params. These should be used sparingly,
    # but in some cases might be needed due to current pipeline implementations,
    # e.g., to ensure we can actually make small models. In general, having to
    # pass extra_args is a code smell for that model and should be fixed since
    # custom models for the corresponding arch created by users may hit the same issues
    extra_args: dict | None = None

    marks: list[MarkDecorator] | None = None


### Create the Omni model for a given acceleration type
# TODO - this should also probably consider the device cost,
# and only collapse down what we actually can for the current
# environment...
ACC_TYPE_MAP: dict = {
    DiffAccs.HSDP: dac_build.build_omni_for_hsdp,
    DiffAccs.TEA_CACHE: dac_build.build_omni_for_teacache,
    DiffAccs.CACHE_DIT: dac_build.build_omni_for_cache_dit,
    DiffAccs.SEQUENCE_PARALLEL: dac_build.build_omni_for_sequence_parallel,
    DiffAccs.CFG_PARALLEL: dac_build.build_omni_for_cfg_parallel,
    DiffAccs.TENSOR_PARALLEL: dac_build.build_omni_for_tensor_parallel,
    DiffAccs.CPU_OFFLOAD: dac_build.build_omni_for_cpu_offload,
    DiffAccs.VAE_PATCH_PARALLEL: dac_build.build_omni_for_patch_parallel,
}


def build_omni_from_acc_type(acc_type: DiffAccs, **kwargs) -> Omni:
    """Given an acceleration type, get the corresponding Omni() kwargs needed
    to enable the acceleration."""
    assert acc_type in ACC_TYPE_MAP
    omni_builder = ACC_TYPE_MAP[acc_type]
    return omni_builder(**kwargs)
