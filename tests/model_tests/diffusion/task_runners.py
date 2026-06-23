"""
Definitions running and validating individual tasks, e.g., text to image,
image to image, and so on. These are called by the core test runner.
"""

import base64
import io

from PIL import Image

from tests.helpers.runtime import dummy_messages_from_mix_data
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPT = "Dummy prompt"
IMAGE_DIMS = (512, 512)
HEIGHT, WIDTH = IMAGE_DIMS
INPUT_IMAGE = Image.new("RGB", IMAGE_DIMS)

# TODO: Add multi-output (n>1) and deterministic (same seed → same output)
# task runners to cover cases currently only tested in e2e expansion tests.

# Offline sampling params
IMAGE_GEN_SAMPLING_PARAMS = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    height=HEIGHT,
    width=WIDTH,
    seed=42,
)

# Online extra_body for diffusion requests
IMAGE_GEN_EXTRA_BODY = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": 4,
    "seed": 42,
}


def run_and_validate_text_to_image_request(omni: Omni):
    """Run a T2I request an ensure the resulting image is valid."""
    outputs = omni.generate(
        {"prompt": PROMPT},
        IMAGE_GEN_SAMPLING_PARAMS,
    )
    validate_image_output(outputs)


def run_and_validate_image_to_image_request(omni: Omni):
    """Run an I2I request an ensure the resulting image is valid."""
    outputs = omni.generate(
        {
            "prompt": PROMPT,
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


### Online (OmniServer) task runners


def _build_online_image_data_url():
    """Encode INPUT_IMAGE as a base64 data URL for online requests."""
    buf = io.BytesIO()
    INPUT_IMAGE.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def run_and_validate_online_text_to_image_request(server, client):
    """Run an online T2I request and ensure the response is valid."""
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": IMAGE_GEN_EXTRA_BODY,
    }
    client.send_diffusion_request(request_config)


def run_and_validate_online_image_to_image_request(server, client):
    """Run an online I2I request and ensure the response is valid."""
    image_data_url = _build_online_image_data_url()
    messages = dummy_messages_from_mix_data(
        content_text=PROMPT,
        image_data_url=image_data_url,
    )
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": IMAGE_GEN_EXTRA_BODY,
    }
    client.send_diffusion_request(request_config)
