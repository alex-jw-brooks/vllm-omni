"""
Definitions running and validating individual tasks, e.g., text to image,
image to image, and so on. These are called by the core test runner.
"""

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

IMAGE_DIMS = (512, 512)
HEIGHT, WIDTH = IMAGE_DIMS
INPUT_IMAGE = Image.new("RGB", IMAGE_DIMS)
IMAGE_GEN_SAMPLING_PARAMS = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    height=HEIGHT,
    width=WIDTH,
    seed=42,
)


def run_and_validate_text_to_image_request(omni: Omni):
    """Run a T2I request an ensure the resulting image is valid."""
    outputs = omni.generate(
        {"prompt": "Dummy prompt"},
        IMAGE_GEN_SAMPLING_PARAMS,
    )
    validate_image_output(outputs)


def run_and_validate_image_to_image_request(omni: Omni):
    """Run an I2I request an ensure the resulting image is valid."""
    outputs = omni.generate(
        {
            "prompt": "Dummy prompt",
            "multi_modal_data": {"image": INPUT_IMAGE},
        },
        IMAGE_GEN_SAMPLING_PARAMS,
    )
    validate_image_output(outputs)


def validate_image_output(outputs):
    """Ensure that an image was produced and that it was the expected size."""
    assert len(outputs) == 1
    images = outputs[0].request_output.images
    assert len(images) == 1
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.size == IMAGE_DIMS


# TODO:
# 1. Add pytest markers to skip test configurations based on device requirements
# 2. Add infra to check for things that are not compatible with each other.
