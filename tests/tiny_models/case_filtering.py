"""
Analogous to: https://github.com/vllm-project/vllm/blob/v0.23.0/tests/models/multimodal/generation/vlm_utils/case_filtering.py

This follows a similar pattern so that we can use `get_parametrized_options` to expand our test
configurations out, similar to the way in which vLLM expands supported input type tests; doing so
lets us essentially do what pytest's parametrization does, but in a way that lets models support
different acceleration types without noisy skip tests for missing acceleration support.
"""

# Engine initialization is currently quite expensive. For efficiency,
# when we run the tests, we should try to collapse compatible accelerations,
# because even with tiny models, the overhead of spinning up a new instance
# stacks up fast.
import itertools

import pytest

from tests.tiny_models.config_types import DiffusionModelTestOpts


def get_model_parametrization(model_name: str, test_info: DiffusionModelTestOpts):
    assert test_info.supported_accelerations is not None
    return [
        pytest.param(
            model_name,
            test_info.builder,
            test_info.extra_args,
            acc,  # TODO: This should actually be an acceleration list or None
            marks=test_info.marks if test_info.marks is not None else [],
        )
        for acc in test_info.supported_accelerations
    ]


def get_parametrized_options(
    test_settings: dict[str, DiffusionModelTestOpts],
):
    """Converts all of our DiffusionModelTestOpts into an expanded list of parameters
    based on which accelerations are available.
    """
    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    parametrization = [
        get_model_parametrization(model_name, test_info) for model_name, test_info in test_settings.items()
    ]
    return list(itertools.chain(*parametrization))
