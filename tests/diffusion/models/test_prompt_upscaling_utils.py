from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import do_prompt_upscaling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def mock_request(extra_args: dict[str, Any]) -> OmniDiffusionRequest:
    return cast(
        OmniDiffusionRequest,
        SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=extra_args),
        ),
    )


def mock_config(enable: bool) -> OmniDiffusionConfig:
    return cast(OmniDiffusionConfig, SimpleNamespace(enable_prompt_upscaling=enable))


@pytest.mark.parametrize(
    "allow_upscale,extra_args,expected",
    [
        (True, {"prompt_upscaling": True}, True),
        (True, {"prompt_upscaling": False}, False),
        (True, {}, False),
        (False, {"prompt_upscaling": True}, False),
        (False, {}, False),
    ],
)
def test_do_prompt_upscaling(allow_upscale, extra_args, expected):
    """Ensure we only do upscale if it's enabled and requested."""
    res = do_prompt_upscaling(mock_request(extra_args), mock_config(allow_upscale))
    assert res is expected


def test_do_prompt_upscaling_rejects_non_bool():
    """Ensure do_prompt_upscaling requires a boolean value."""
    with pytest.raises(TypeError, match="must be a bool"):
        do_prompt_upscaling(mock_request({"prompt_upscaling": object()}), mock_config(True))
