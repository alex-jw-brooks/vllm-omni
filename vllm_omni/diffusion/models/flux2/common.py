# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utils and base class for Flux2 / Flux2 Klein."""

import json
import math
import os
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_flux2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    if od_config.output_type == "latent":
        return lambda x: x
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class Flux2ImageProcessor(VaeImageProcessor):
    """Image processor to preprocess the reference image for Flux2 / Klein."""

    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image,
        max_aspect_ratio: int = 8,
        min_side_length: int = 64,
        max_area: int = 1024 * 1024,
    ) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size
        if width < min_side_length or height < min_side_length:
            raise ValueError(f"Image too small: {width}x{height}. Both dimensions must be at least {min_side_length}px")

        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}x{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        if width * height > max_area:
            logger.warning("Image area exceeds recommended maximum; resizing will be applied.")

        return image

    @staticmethod
    def _resize_to_target_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        scale = math.sqrt(target_area / (image_width * image_height))
        width = int(image_width * scale)
        height = int(image_height * scale)
        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)


# Flux2 / Flux2 Klein share the same feature support at the moment, largely because they share the DiT architecture
class Flux2PipelineBase(nn.Module, CFGParallelMixin, SupportImageInput, DiffusionPipelineProfilerMixin):
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
        ).to(self._execution_device)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, Flux2Transformer2DModel)
        self.transformer = Flux2Transformer2DModel(quant_config=od_config.quantization_config, **transformer_kwargs)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # Subclasses must set this to their model-specific layer indices
    _default_text_encoder_out_layers: tuple[int, ...] = ()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return False

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        **kwargs,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                "`height` and `width` have to be divisible by %s but are %s and %s. "
                "Dimensions will be resized accordingly",
                self.vae_scale_factor * 2,
                height,
                width,
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    def _get_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None,
        max_sequence_length: int,
        hidden_states_layers: tuple[int, ...],
    ):
        raise NotImplementedError("Flux2 subclasses have different implementations for embedding prompts")

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None,
        max_sequence_length: int,
        text_encoder_out_layers: tuple[int, ...],
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_image_ids
    def _prepare_image_ids(
        image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):
        r"""
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (List[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """

        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._pack_latents
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpack_latents_with_ids
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_latent_ids
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        layer_ids = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, layer_ids)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_text_ids
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: torch.Tensor | None = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            seq_positions = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, seq_positions)
            out_ids.append(coords)

        return torch.stack(out_ids)

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_image_latents
    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(image_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def interrupt(self):
        return self._interrupt

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # Record components loaded by diffusers submodules to satisfy strict checks.
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights

    def _denoise(
        self,
        *,
        latents,
        latent_ids,
        image_latents,
        image_latent_ids,
        prompt_embeds,
        text_ids,
        negative_prompt_embeds,
        negative_text_ids,
        timesteps,
        guidance_scale,
    ):
        raise NotImplementedError

    def _decode_latents(self, latents, latent_ids, output_type):
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            if latents.dtype != self.vae.dtype:
                latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def _extract_images(self, first_prompt) -> list[PIL.Image.Image] | None:
        """Extract and validate reference images from a request prompt."""
        if isinstance(first_prompt, str):
            return None
        raw_image = first_prompt.get("multi_modal_data", {}).get("image")
        if raw_image is None:
            return None
        if not isinstance(raw_image, list):
            raw_image = [raw_image]
        return [PIL.Image.open(im) if isinstance(im, str) else im for im in raw_image]

    def _preprocess_condition_images(
        self,
        images: list[PIL.Image.Image],
        height: int | None,
        width: int | None,
    ) -> tuple[list[torch.Tensor], int | None, int | None]:
        """Validate, resize, and preprocess reference images for VAE encoding."""
        condition_images = []
        for img in images:
            self.image_processor.check_image_input(img)
            img = self.image_processor._resize_to_target_area(img) if img.size[0] * img.size[1] > 1024 * 1024 else img
            image_width, image_height = img.size

            multiple_of = self.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_images.append(img)
            height = height or image_height
            width = width or image_width
        return condition_images, height, width

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request. Taking only the first prompt for now."
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        req_prompt_embeds = [p.get("prompt_embeds") if not isinstance(p, str) else None for p in req.prompts]

        if any(p is not None for p in req_prompt_embeds):
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore # intentionally expect TypeError
        else:
            prompt_embeds = None

        # Apply defaults, then override with request sampling parameters
        ### FIXME - we need to pass the correct overrides through here.
        num_inference_steps = req.sampling_params.num_inference_steps or 50
        guidance_scale = req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else 4.0
        output_type = req.sampling_params.output_type or "pil"
        max_sequence_length = req.sampling_params.max_sequence_length or 512
        text_encoder_out_layers = req.sampling_params.extra_args.get(
            "text_encoder_out_layers",
            self._default_text_encoder_out_layers,
        )

        height = req.sampling_params.height
        width = req.sampling_params.width

        req_sigmas = req.sampling_params.sigmas
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if req_sigmas is None else req_sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        generator = req.sampling_params.generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )

        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Batch size
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:  # Presumably prompt is None & str
            batch_size = 1

        device = self._execution_device

        # 3. Prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. Negative prompt embeddings (only for Klein, which supports CFG at the moment)
        negative_prompt_embeds = None
        negative_text_ids = None
        if self.do_classifier_free_guidance:
            req_neg = [p.get("negative_prompt_embeds") if not isinstance(p, str) else None for p in req.prompts]
            neg_embeds_input = torch.stack(req_neg) if any(p is not None for p in req_neg) else None  # type: ignore

            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)

            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=neg_embeds_input,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 5. Process reference images (if provided for image editing)
        images = self._extract_images(first_prompt)
        condition_images = None
        if images is not None:
            condition_images, height, width = self._preprocess_condition_images(images, height, width)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 7. Prepare timesteps
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # 8. Run the corresponding _denoise loop; this will vary depending
        # on whether or not the subclass supports CFG or not.
        latents = self._denoise(
            latents=latents,
            latent_ids=latent_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
        )
        self._current_timestep = None

        # 8. Decode
        return self._decode_latents(latents, latent_ids, output_type)
