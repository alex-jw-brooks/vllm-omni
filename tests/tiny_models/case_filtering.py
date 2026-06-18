"""
Analogous to: https://github.com/vllm-project/vllm/blob/v0.23.0/tests/models/multimodal/generation/vlm_utils/case_filtering.py

This follows a similar pattern so that we can use `get_parametrized_options` to expand our test
configurations out, similar to the way in which vLLM expands supported input type tests. For now,
this mostly just means parametrizing over the test groups, which may be unique per model, but
doing so in this way lets us cleanly define test groups that may be heterogeneous + avoid having
excessive skip marks for features that may not be supported in every pipeline.
"""

# Engine initialization is currently quite expensive. For efficiency,
# when we run the tests, we should try to collapse compatible accelerations,
# because even with tiny models, the overhead of spinning up a new instance
# stacks up fast.
import itertools

import pytest

from tests.tiny_models.config_types import DiffusionModelTestOpts, get_required_device_count
from vllm_omni.platforms import current_omni_platform


def get_test_group_marks(test_group, model_marks: list | None) -> list:
    """Build the full set of pytest marks for a test group. This will append a skip
    based on the device counts for the current platform if we don't have the device
    count needed to run the test group."""
    marks = list(model_marks) if model_marks is not None else []
    required_devices = get_required_device_count(test_group)
    if required_devices > 1:
        assert current_omni_platform is not None and current_omni_platform.device_count is not None
        device_count = current_omni_platform.device_count()
        if device_count < required_devices:
            marks.append(
                pytest.mark.skip(
                    reason=f"Need {required_devices} devices, got {device_count}",
                )
            )
    return marks


def get_model_parametrization(model_name: str, test_info: DiffusionModelTestOpts):
    """Given a model & its corresponding test options, build the list of pytest params
    to be run for this model. For now, this just means running over the test groups, but
    writing it this way for now in case we need additionally flexibility later (e.g., could
    build test groups more dynamically based on number of visible devices, etc).
    """
    assert test_info.test_groups is not None
    return [
        pytest.param(
            model_name,
            test_info.builder,
            test_group,
            test_info.supported_tasks,
            marks=get_test_group_marks(test_group, test_info.marks),
        )
        for test_group in test_info.test_groups
    ]


def get_parametrized_options(
    test_settings: dict[str, DiffusionModelTestOpts],
):
    """Converts all the DiffusionModelTestOpts into an expanded list of parameters
    based on which accelerations are available.
    """
    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    parametrization = [
        get_model_parametrization(model_name, test_info) for model_name, test_info in test_settings.items()
    ]
    return list(itertools.chain(*parametrization))
