import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
)
from tests.utils import hardware_marks

TWO_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=2)
FOUR_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)
POSITIVE_PROMPT = "A cat sitting on a windowsill"
NEGATIVE_PROMPT = "blurry, low quality"


# Currently Flux2 tests target Flux2 Klein.
def _get_diffusion_feature_cases(model: str, gguf_model: str):
    return [
        # CPU offload / HSDP
        pytest.param(
            OmniServerParams(
                model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-cpu-offload",
                    "--use-hsdp",
                    "--hsdp-shard-size",
                    "2",
                ],
            ),
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # FP8 / Hybrid sequence parallelism
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring-degree",
                    "2",
                    "--ulysses-degree",
                    "2",
                    "--quantization",
                    "fp8",
                ],
            ),
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
        # GGUF / TP / CFG parallel
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--quantization-config",
                    f'{{"method":"gguf","gguf_model":"{gguf_model}"}}',
                    "--cache-backend",
                    "cache_dit",
                    "--cfg-parallel-size",
                    "2",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(
        model="black-forest-labs/FLUX.2-klein-4B",
        gguf_model="unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf",
    ),
    indirect=True,
)
def test_flux2_klein(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    request_config = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": POSITIVE_PROMPT}],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
