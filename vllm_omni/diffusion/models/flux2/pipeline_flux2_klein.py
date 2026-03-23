# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import os

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.flux2.common import (
    Flux2PipelineBase,
    compute_empirical_mu,
    get_flux2_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Flux2KleinPipeline(Flux2PipelineBase):
    """Flux2 klein pipeline for text-to-image generation."""

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        is_distilled: bool = False,
    ):
        super().__init__(od_config=od_config, prefix=prefix)
        self.is_distilled = is_distilled

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        ).to(self._execution_device)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

    def _get_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None,
        max_sequence_length: int,
        hidden_states_layers: tuple[int, ...],
    ):
        """Get the prompt embeddings for Qwen3."""
        dtype = self.text_encoder.dtype
        device = self.text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    def check_inputs(
        self,
        *args,
        guidance_scale=None,
        **kwargs,
    ):
        super().check_inputs(*args, **kwargs)

        if guidance_scale is not None and guidance_scale > 1.0 and self.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1 and not self.is_distilled

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        # NOTE: For now, flux2 klein pipeline doesn't support image edit

        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request.Taking only the first image for now."
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        req_prompt_embeds = [p.get("prompt_embeds") if not isinstance(p, str) else None for p in req.prompts]

        if any(p is not None for p in req_prompt_embeds):
            # If at list one prompt is provided as an embedding,
            # Then assume that the user wants to provide embeddings for all prompts, and enter this if block
            # If the user in fact provides mixed input format, req_prompt_embeds will have some None's
            # And `torch.stack` automatically raises an exception for us
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore # intentionally expect TypeError
        else:
            prompt_embeds = None

        req_negative_prompt_embeds = [
            p.get("negative_prompt_embeds") if not isinstance(p, str) else None for p in req.prompts
        ]
        if any(p is not None for p in req_negative_prompt_embeds):
            negative_prompt_embeds = torch.stack(req_negative_prompt_embeds)  # type: ignore # intentionally expect TypeError
        else:
            negative_prompt_embeds = None

        # Apply the default overrides for Flux2Kleins's sampling parameters
        # TODO (Alex) we need to expose these more generically
        num_inference_steps = 50
        guidance_scale = 4.0
        output_type = "pil"
        max_sequence_length = 512
        text_encoder_out_layers = (9, 18, 27)

        height = req.sampling_params.height
        width = req.sampling_params.width
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        output_type = req.sampling_params.output_type or output_type
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        text_encoder_out_layers = req.sampling_params.extra_args.get(
            "text_encoder_out_layers",
            text_encoder_out_layers,
        )

        req_sigmas = req.sampling_params.sigmas
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if req_sigmas is None else req_sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        generator = req.sampling_params.generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:  # Presumably prompt is None & str
            batch_size = 1

        device = self._execution_device

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 4. prepare latent variables
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

        # 5. Prepare timesteps
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

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(self.transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": None,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "joint_attention_kwargs": self.attention_kwargs,
                "return_dict": False,
            }
            if self.do_classifier_free_guidance:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep / 1000,
                    "guidance": None,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": self.attention_kwargs,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            # For editing pipelines, we need to slice the output to remove condition latents
            output_slice = latents.size(1) if image_latents is not None else None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=self.do_classifier_free_guidance,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
                output_slice=output_slice,
            )

            # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, self.do_classifier_free_guidance)

        self._current_timestep = None

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


# For now, explicitly re-export the shared flux2 to play nicely
# with the existing patterns
__all__ = [
    "Flux2KleinPipeline",
    "get_flux2_post_process_func",
]
