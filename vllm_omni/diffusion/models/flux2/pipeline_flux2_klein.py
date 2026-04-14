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

import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.flux2.common import (
    Flux2PipelineBase,
    get_flux2_post_process_func,
)

logger = init_logger(__name__)


class Flux2KleinPipeline(Flux2PipelineBase):
    """Flux2 klein pipeline for text-to-image generation."""

    _default_text_encoder_out_layers = (9, 18, 27)

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

        self._text_encoder = Qwen3ForCausalLM.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        ).to(self._execution_device)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

    @property
    def text_encoder(self) -> torch.nn.Module:
        """Text encoder implementation for Flux2 Klein (Qwen3)."""
        return self._text_encoder

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
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale is not None and self._guidance_scale > 1 and not self.is_distilled

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
        """Runs the denoising loop for Flux2 Klein (supports CFG)."""
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
                "joint_attention_kwargs": {},
            }
            if self.do_classifier_free_guidance:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep / 1000,
                    "guidance": None,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": {},
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
        return latents


# For now, explicitly re-export the shared flux2 to play nicely
# with the existing patterns
__all__ = [
    "Flux2KleinPipeline",
    "get_flux2_post_process_func",
]
