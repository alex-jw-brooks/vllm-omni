# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for /v1/images/generations with per-request LoRA.

This validates:
- The API server accepts a per-request `lora` object in the Images API payload.
- LoRA can be switched per request (adapter A -> adapter B -> no LoRA).
- Output correctness is asserted using a small image slice with tolerance.
"""

import base64
import json
import os
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from tests.conftest import OmniServerParams, OpenAIClientHandler
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Increase timeout for downloading assets from S3 (default 5s is too short for CI)
os.environ.setdefault("VLLM_IMAGE_FETCH_TIMEOUT", "60")

MODEL = "Tongyi-MAI/Z-Image-Turbo"
DIFFUSION_INIT_TIMEOUT_S = 700
PROMPT = "a photo of a cat sitting on a laptop keyboard"
SIZE = "256x256"
SEED = 42

z_image_params = [
    OmniServerParams(
        model=MODEL,
        server_args=[
            "--num-gpus",
            "1",
            "--stage-init-timeout",
            str(DIFFUSION_INIT_TIMEOUT_S),
            "--init-timeout",
            str(DIFFUSION_INIT_TIMEOUT_S),
        ],
    )
]


### End to end tests
def _generate_image(
    openai_client: OpenAIClientHandler,
    model: str,
    prompt: str = PROMPT,
    size: str = SIZE,
    extra_body: dict | None = None,
    timeout: int = 900,
) -> Image.Image:
    response = openai_client.client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size=size,
        response_format="b64_json",
        extra_body={
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": SEED,
            **(extra_body or {}),
        },
        timeout=timeout,
    )
    img_bytes = base64.b64decode(response.data[0].b64_json)
    img = Image.open(BytesIO(img_bytes))
    img.load()
    return img.convert("RGB")


@pytest.mark.parametrize("omni_server", z_image_params, indirect=True)
def test_t2i_concurrent_requests_different_sizes(omni_server, openai_client) -> None:
    """Test /v1/images/generations concurrent requests with different sizes."""
    barrier = threading.Barrier(2)
    results: list[tuple[int, int]] = []

    def _call_generate(size: str) -> None:
        barrier.wait()
        img = _generate_image(
            openai_client,
            omni_server.model,
            prompt="cute cat playing with a ball",
            size=size,
            timeout=120,
        )
        results.append(img.size)

    threads = [
        threading.Thread(target=_call_generate, args=("512x512",)),
        threading.Thread(target=_call_generate, args=("768x512",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert (512, 512) in results
    assert (768, 512) in results


### LoRA related
def _write_zimage_lora(adapter_dir: Path, *, q_scale: float = 0.0, k_scale: float = 0.0, v_scale: float = 0.0):
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Z-Image transformer uses dim=3840 by default.
    dim = 3840
    module_name = "transformer.layers.0.attention.to_qkv"
    rank = 1

    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, 0] = 1.0

    # QKVParallelLinear packs (Q, K, V) => out dim is 3 * dim (tp=1).
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    if q_scale:
        lora_b[:dim, 0] = q_scale
    if k_scale:
        lora_b[dim : 2 * dim, 0] = k_scale
    if v_scale:
        lora_b[2 * dim :, 0] = v_scale

    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": rank,
                "target_modules": [module_name],
            }
        ),
        encoding="utf-8",
    )


def _image_blue_tail_slice(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    assert arr.ndim == 3 and arr.shape[-1] == 3
    tail = arr[-3:, -3:, -1].astype(np.float32)
    assert tail.shape == (3, 3)
    return tail


def _slice_diff_stats(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(actual - expected)
    return float(diff.max()), float(diff.mean())


def _assert_slice_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    base_max: float,
    base_mean: float,
) -> None:
    assert actual.shape == (3, 3)
    assert expected.shape == (3, 3)
    max_diff, mean_diff = _slice_diff_stats(actual, expected)
    # NOTE: Different attention backends / torch.compile can introduce small
    # floating-point drift that shows up as a few LSBs in uint8 pixels. Keep
    # the reset check tolerant but bounded to avoid flaky CI.
    max_thresh = max(10.0, base_max + 4.0)
    mean_thresh = max(6.0, base_mean + 4.0)
    assert max_diff <= max_thresh and mean_diff <= mean_thresh, (
        f"{label} slice mismatch (max={max_diff:.1f} > {max_thresh:.1f} or "
        f"mean={mean_diff:.1f} > {mean_thresh:.1f}): {actual.tolist()}"
    )


def _assert_slice_diff(actual: np.ndarray, baseline: np.ndarray, *, label: str) -> None:
    assert actual.shape == (3, 3)
    assert baseline.shape == (3, 3)
    diff = np.abs(actual - baseline).mean()
    assert diff > 0.1, f"{label} slice diff too small: {diff} ({actual.tolist()} vs {baseline.tolist()})"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("omni_server", z_image_params, indirect=True)
def test_images_generations_per_request_lora_switching(
    omni_server, tmp_path: Path, openai_client: OpenAIClientHandler
) -> None:
    # Base generation.
    base_img = _generate_image(openai_client, omni_server.model)
    base_slice = _image_blue_tail_slice(base_img)
    base_ref_img = _generate_image(openai_client, omni_server.model)
    base_ref_slice = _image_blue_tail_slice(base_ref_img)
    base_ref_max, base_ref_mean = _slice_diff_stats(base_ref_slice, base_slice)

    # Adapter A: apply delta to V slice only.
    lora_a_dir = tmp_path / "zimage_lora_a"
    _write_zimage_lora(lora_a_dir, v_scale=8.0)
    lora_a = {"name": "a", "path": str(lora_a_dir), "scale": 64.0}
    img_a = _generate_image(openai_client, omni_server.model, extra_body={"lora": lora_a})
    a_slice = _image_blue_tail_slice(img_a)
    _assert_slice_diff(a_slice, base_slice, label="lora_a_vs_base")
    a_vs_base = float(np.abs(a_slice - base_slice).mean())

    # Adapter B: apply delta to K slice only (should differ from adapter A).
    lora_b_dir = tmp_path / "zimage_lora_b"
    _write_zimage_lora(lora_b_dir, k_scale=4.0)
    lora_b = {"name": "b", "path": str(lora_b_dir), "scale": 64.0}
    img_b = _generate_image(openai_client, omni_server.model, extra_body={"lora": lora_b})
    b_slice = _image_blue_tail_slice(img_b)
    _assert_slice_diff(b_slice, base_slice, label="lora_b_vs_base")
    _assert_slice_diff(b_slice, a_slice, label="lora_b_vs_lora_a")
    b_vs_base = float(np.abs(b_slice - base_slice).mean())
    b_vs_a = float(np.abs(b_slice - a_slice).mean())

    # Ensure switching back to no-LoRA restores the base output.
    base_img_2 = _generate_image(openai_client, omni_server.model)
    base_slice_2 = _image_blue_tail_slice(base_img_2)
    _, base_reset_mean = _slice_diff_stats(base_slice_2, base_slice)
    _assert_slice_close(
        base_slice_2,
        base_slice,
        label="base_after_reset",
        base_max=base_ref_max,
        base_mean=base_ref_mean,
    )

    # Ensure LoRA effects are clearly above the baseline drift.
    min_delta = max(base_reset_mean + 1.0, 1.5)
    assert a_vs_base > min_delta, f"lora_a_vs_base drift too small: {a_vs_base} <= {min_delta}"
    assert b_vs_base > min_delta, f"lora_b_vs_base drift too small: {b_vs_base} <= {min_delta}"
    assert b_vs_a > min_delta, f"lora_b_vs_lora_a drift too small: {b_vs_a} <= {min_delta}"
