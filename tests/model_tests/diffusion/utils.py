"""
Utilities for resolving real models to their tiny model equivalents.
"""

import logging

from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from vllm_omni.diffusion.data import resolve_model_class_name

logger = logging.getLogger(__name__)


def resolve_tiny_model_path(model: str) -> str:
    """Given a real model name/path, resolve it to a tiny model path.

    Returns the original model path if no suitable tiny builder is found."""
    pipeline_class = resolve_model_class_name(model)
    if pipeline_class is None:
        # resolve_model_class_name is currently diffusion only, but this is also integrated
        # into the non-common tests so that the tiny builders can be used for diffusion e2e
        # tests. If we can't find a pipeline_class, it is most likely because the test is
        # for a non-diffusion model.
        logger.warning(
            "Could not resolve the pipeline config for %s; is it a diffusion model?",
            model,
        )
        return model

    test_opts = DIFFUSION_TEST_SETTINGS.get(pipeline_class)
    if test_opts is None:
        logger.warning(
            "No tiny model builder for pipeline %s (model: %s). Using original model.",
            pipeline_class,
            model,
        )
        return model

    return test_opts.builder()
