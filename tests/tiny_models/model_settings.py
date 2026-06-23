from tests.tiny_models import diff_model_builders
from tests.tiny_models.config_types import (
    DiffAccs,
    DiffTasks,
    DiffusionModelTestOpts,
)

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
# $ pytest test_common_offline.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline
#   ^ Runs all 4 test_groups for Flux2KleinPipeline only
#
# $ pytest test_common_offline.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline]
#   ^ Runs only the case with no accelerations enabled for only Flux2KleinPipeline
#
# $ pytest test_common_offline.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline[
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
