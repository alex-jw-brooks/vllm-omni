# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

import torch
from diffusers.pipelines.flux2.system_messages import (
    SYSTEM_MESSAGE,
)
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.flux2.common import (
    Flux2PipelineBase,
    get_flux2_post_process_func,
)

logger = init_logger(__name__)


# Adapted from diffusers.pipelines.flux2.pipeline_flux2.format_input
def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
) -> list[list[dict[str, Any]]]:
    """
    Format a batch of text prompts into the conversation format expected by apply_chat_template.

    Args:
        prompts: List of text prompts
        system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)

    Returns:
        `list[list[dict[str, Any]]]`: List of conversations, where each conversation is a list of message dicts
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    # Currently we assumes images is None
    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


class Flux2Pipeline(Flux2PipelineBase):
    """Flux2 pipeline for text-to-image generation."""

    _default_text_encoder_out_layers = (10, 20, 30)

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config, prefix=prefix)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self._execution_device)
        self.tokenizer = PixtralProcessor.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )

        self.system_message = SYSTEM_MESSAGE

    def _get_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None,
        max_sequence_length: int,
        hidden_states_layers: tuple[int, ...],
    ):
        """Get the prompt embeddings for Mistral 3 small."""
        dtype = self.text_encoder.dtype
        device = self.text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_input(
            prompts=prompt,
            system_message=self.system_message,
        )

        # Process all messages at once
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

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
        """Runs the denoising loop for Flux2 (does not use CFG)."""
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

            noise_pred = self.transformer(
                hidden_states=latent_model_input,  # (B, image_seq_len, C)
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,  # B, text_seq_len, 4
                img_ids=latent_image_ids,  # B, image_seq_len, 4
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

            noise_pred = noise_pred[:, : latents.size(1)]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)
        return latents


# For now, explicitly re-export the shared flux2 to play nicely
# with the existing patterns
__all__ = [
    "Flux2Pipeline",
    "get_flux2_post_process_func",
]
