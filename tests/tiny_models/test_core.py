from shutil import rmtree

import pytest

from tests.tiny_models import diff_model_builders
from tests.tiny_models.case_filtering import get_parametrized_options
from tests.tiny_models.config_types import (
    DiffAccs,
    DiffTasks,
    DiffusionModelTestOpts,
    build_omni_from_diff_accelerations,
)
from tests.tiny_models.task_runners import (
    run_and_validate_image_to_image_request,
    run_and_validate_text_to_image_request,
)


# NOTE: Pipelines are not consistent in handling input multimodal data
# right now, so for image2image, we have to pass a PIL image.
# TODO (Alex): Standardize input types & add common checks
@pytest.fixture(scope="session")
def tiny_model_paths(request):
    """Build or download the tiny models for the selected tests.

    NOTE: this is session scoped to avoid churn in tiny model creation,
    but will ensure all the tiny models you need are created for the selected tests
    before it starts to execute them. If you exclude tiny models"""
    model_paths = {}
    print("Initializing tiny models...")
    for item in request.session.items:
        if not hasattr(item, "callspec"):
            raise ValueError("tiny_model_paths should not be used with non-parametrized models.")
        model_name = item.callspec.params["model_name"]
        if model_name not in model_paths:
            print(f"Calling tiny model builder for: {model_name}")
            model_builder = item.callspec.params["model_builder"]
            model_paths[model_name] = model_builder()

    yield model_paths
    for path in model_paths.values():
        rmtree(path, ignore_errors=True)


# This object defines the (tiny) model configurations for common tests.
#
# TO ADD A NEW MODEL:
# you should map the pipeline name (i.e., the name of the pipeline in _DIFFUSION_MODELS)
# to a DiffusionModelTestOpts and define the supported tasks. Then, configure the test_groups,
# which should be None (no acceleration) or a list of accelerations to run together.
#
# This is mostly done for efficiency, since these are largely smoke tests to make sure
# the model loads and produces an output of the right shape without exploding. The model
# will only be loaded once per test group, and will execute each of its supported tasks
# as a pytest subtest.
#
# TO RUN A SUBSET OF THE TESTS:
# These tests should be fast, but if you'd like to run a subset of the tests, you can do so
# with `-k`; the IDs of the tests are the name of the pipeline joined by +.
# Examples:
# $ pytest test_core.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline
#   ^ Runs all 4 test_groups for Flux2KleinPipeline only
#
# $ pytest test_core.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline]
#   ^ Runs only the case with no accelerations enabled for only Flux2KleinPipeline
#
# $ pytest test_core.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline[
#   ^ Runs only the case with at least one acceleration enabled for only Flux2KleinPipeline
DIFFUSION_TEST_SETTINGS = {
    "Flux2KleinPipeline": DiffusionModelTestOpts(
        supported_tasks=[DiffTasks.TEXT_TO_IMAGE, DiffTasks.IMAGE_EDIT],
        builder=diff_model_builders.tiny_flux2_klein_builder,
        test_groups=[
            None,
            [DiffAccs.HSDP, DiffAccs.TEA_CACHE],
            [DiffAccs.SEQUENCE_PARALLEL, DiffAccs.CACHE_DIT],
            [DiffAccs.CFG_PARALLEL, DiffAccs.TENSOR_PARALLEL],
        ],
    )
}


@pytest.mark.parametrize(
    "model_name,accelerations,supported_tasks,model_builder",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_pipeline_on_supported_tasks(
    model_name,
    accelerations: list[DiffAccs] | None,
    model_builder,
    supported_tasks: list[DiffTasks],
    tiny_model_paths,
    subtests,
):
    """Run a smoke test on all of the pipelines supported tasks using a set of enabled accelerations."""
    assert len(supported_tasks) > 0
    # We initialize the Omni object before running the tasks, then run each task as a pytest subtask.
    # This lets us init the model once, but display separate failures in pytest, and avoid halting the
    # checks on other tasks if one fails.
    #
    # This allows ut so have some degree of test isolation without the cost of redundant initialization,
    # since starting the server can take 10+ seconds, even for tiny models.
    #
    # NOTE: Be sure to install pytest-subtests if you're running on pytest < 9
    omni = build_omni_from_diff_accelerations(
        accelerations=accelerations,
        model=tiny_model_paths[model_name],
        enforce_eager=True,
    )

    for task_type in supported_tasks:
        with subtests.test(msg=task_type):
            if task_type == DiffTasks.TEXT_TO_IMAGE:
                run_and_validate_text_to_image_request(omni)
            elif task_type == DiffTasks.IMAGE_EDIT:
                run_and_validate_image_to_image_request(omni)
            else:
                raise ValueError(f"Task type {task_type} is not yet supported")
