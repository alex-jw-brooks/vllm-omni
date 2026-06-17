from shutil import rmtree

import pytest
from PIL import Image

from tests.tiny_models import diff_model_builders
from tests.tiny_models.case_filtering import get_parametrized_options
from tests.tiny_models.config_types import DiffAccs, DiffTasks, DiffusionModelTestOpts, build_omni_from_acc_type
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# NOTE: Pipeline are not consistent in handling input multimodal data
# right now, so for image2image, we have to pass a PIL image.
# TODO (Alex): Standardize input types & add common checks
IMAGE_DIMS = (512, 512)
HEIGHT, WIDTH = IMAGE_DIMS
INPUT_IMAGE = Image.new("RGB", IMAGE_DIMS)

DIFFUSION_TEST_SETTINGS = {
    "Flux2KleinPipeline": DiffusionModelTestOpts(
        # Indicates this pipeline implementation actually supports
        # both tti and i2i, so we should run both tests on it
        supported_tasks=[DiffTasks.TEXT_TO_IMAGE, DiffTasks.IMAGE_EDIT],
        builder=diff_model_builders.tiny_flux2_klein_builder,
        supported_accelerations=[
            DiffAccs.HSDP,
            DiffAccs.TEA_CACHE,
            DiffAccs.CACHE_DIT,
            DiffAccs.SEQUENCE_PARALLEL,
            DiffAccs.CFG_PARALLEL,
            DiffAccs.TENSOR_PARALLEL,
        ],
    )
}


@pytest.mark.parametrize(
    "model_name,model_builder,extra_args,acceleration_type",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_text_to_image(model_name, model_builder, extra_args, acceleration_type: DiffAccs):
    # TODO - handle model builder more cleanly, there is really no reason it
    # should be handled here, we can handle it the parametrized expansion
    tiny_model_path = model_builder()
    run_image_generation_test(
        acceleration_type=acceleration_type,
        model_path=tiny_model_path,
        extra_args=extra_args if extra_args is not None else {},
        execution_func=run_and_validate_text_to_image_request,
    )
    rmtree(tiny_model_path)


@pytest.mark.parametrize(
    "model_name,model_builder,extra_args,acceleration_type",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_image_edit(model_name, model_builder, extra_args, acceleration_type: DiffAccs):
    # TODO - handle model builder more cleanly, there is really no reason it
    # should be handled here, we can handle it the parametrized expansion
    tiny_model_path = model_builder()
    run_image_generation_test(
        acceleration_type=acceleration_type,
        model_path=tiny_model_path,
        extra_args=extra_args if extra_args is not None else {},
        execution_func=run_and_validate_image_to_image_request,
    )
    rmtree(tiny_model_path)


def run_image_generation_test(acceleration_type: DiffAccs, model_path, extra_args, execution_func):
    """Common flow for initializing an offline Omni instance, running a dummy image generation request,
    and validating that the output is nonempty & the right shape."""
    print(f"Building for acceleration {acceleration_type}")
    omni = build_omni_from_acc_type(
        acc_type=acceleration_type,
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
# 2. Port mark parametrize infra from vLLM to dynamic cross product the tests
# 3. Common fixtures to only load the model once (this may need to be revisited....)
# 4. Skip tests based on device requirements
# 5. Add pytest marks, ability to filter what we actually run
# 6. Move the tempdir stuff to be handled by pytest (session scoped)
# 7. Write alignment tests
