from shutil import rmtree

import pytest
from PIL import Image

from tests.tiny_models import diff_model_builders
from tests.tiny_models.case_filtering import get_parametrized_options
from tests.tiny_models.config_types import (
    DiffAccs,
    DiffTasks,
    DiffusionModelTestOpts,
    build_omni_from_diff_accelerations,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


# NOTE: Pipeline are not consistent in handling input multimodal data
# right now, so for image2image, we have to pass a PIL image.
# TODO (Alex): Standardize input types & add common checks
@pytest.fixture(scope="session")
def tiny_model_paths(request):
    """Build or download the tiny models for the selected tests."""
    model_paths = {}
    print("Initializing...")
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


IMAGE_DIMS = (512, 512)
HEIGHT, WIDTH = IMAGE_DIMS
INPUT_IMAGE = Image.new("RGB", IMAGE_DIMS)

DIFFUSION_TEST_SETTINGS = {
    "Flux2KleinPipeline": DiffusionModelTestOpts(
        # Indicates this pipeline implementation supports both tti and i2i.
        # In this case, we run both the smoke tests for TTI and for Image edit
        # together to avoid loading the model twice, because doing so is slow.
        supported_tasks=[DiffTasks.TEXT_TO_IMAGE, DiffTasks.IMAGE_EDIT],
        builder=diff_model_builders.tiny_flux2_klein_builder,
        test_groups=[
            None,  # No accelerations
            [DiffAccs.HSDP, DiffAccs.TEA_CACHE],
            [DiffAccs.SEQUENCE_PARALLEL, DiffAccs.CACHE_DIT],
            [DiffAccs.CFG_PARALLEL, DiffAccs.TENSOR_PARALLEL],
        ],
    )
}


@pytest.mark.parametrize(
    "model_name,model_builder,extra_args,accelerations",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_text_to_image(model_name, model_builder, extra_args, accelerations: list[DiffAccs] | None, tiny_model_paths):
    run_image_generation_test(
        accelerations=accelerations,
        model_path=tiny_model_paths[model_name],
        extra_args=extra_args if extra_args is not None else {},
        execution_func=run_and_validate_text_to_image_request,
    )


@pytest.mark.parametrize(
    "model_name,model_builder,extra_args,accelerations",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_image_edit(model_name, model_builder, extra_args, accelerations: list[DiffAccs] | None, tiny_model_paths):
    run_image_generation_test(
        accelerations=accelerations,
        model_path=tiny_model_paths[model_name],
        extra_args=extra_args if extra_args is not None else {},
        execution_func=run_and_validate_image_to_image_request,
    )


def run_image_generation_test(accelerations: list[DiffAccs] | None, model_path, extra_args, execution_func):
    """Common flow for initializing an offline Omni instance, running a dummy image generation request,
    and validating that the output is nonempty & the right shape."""
    print(f"Building for accelerations: {accelerations}")
    omni = build_omni_from_diff_accelerations(
        accelerations=accelerations,
        model=model_path,
        enforce_eager=True,
    )
    execution_func(omni, extra_args)


def run_and_validate_text_to_image_request(omni: Omni, extra_args: dict):
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=4,
        height=HEIGHT,
        width=WIDTH,
        seed=42,
        extra_args=extra_args,
    )
    outputs = omni.generate(
        {"prompt": "Dummy prompt"},
        sampling_params,
    )
    validate_image_output(outputs)


def run_and_validate_image_to_image_request(omni: Omni, extra_args: dict):
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=4,
        height=HEIGHT,
        width=WIDTH,
        seed=42,
        extra_args=extra_args,
    )
    outputs = omni.generate(
        {
            "prompt": "Dummy prompt",
            "multi_modal_data": {"image": INPUT_IMAGE},
        },
        sampling_params,
    )
    validate_image_output(outputs)


def validate_image_output(outputs):
    assert len(outputs) == 1
    images = outputs[0].request_output.images
    assert len(images) == 1
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.size == IMAGE_DIMS


# TODO:
# 4. Skip tests based on device requirements
# 5. Add pytest marks, ability to filter what we actually run
# 7. Write alignment tests
