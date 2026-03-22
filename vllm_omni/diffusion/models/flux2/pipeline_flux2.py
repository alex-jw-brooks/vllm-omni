# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import UPSAMPLING_MAX_IMAGE_SIZE
from diffusers.pipelines.flux2.system_messages import (
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from transformers import AutoProcessor, Mistral3ForConditionalGeneration, PixtralProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux2.common import (
    Flux2ImageProcessor,
    Flux2PipelineBase,
    compute_empirical_mu,
    get_flux2_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)
get_flux2_post_process_func = get_flux2_post_process_func


# Copied from diffusers.pipelines.flux2.pipeline_flux2.format_input
def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
    images: list[PIL.Image.Image] | list[list[PIL.Image.Image]] = None,
) -> list[list[dict[str, Any]]]:
    """
    Format a batch of text prompts into the conversation format expected by apply_chat_template. Optionally, add images
    to the input.

    Args:
        prompts: List of text prompts
        system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
        images (optional): List of images to add to the input.

    Returns:
        `list[list[dict[str, Any]]]`: List of conversations, where each conversation is a list of message dicts
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
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
    else:
        assert len(images) == len(prompts), "Number of images must match number of prompts"
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            ]
            for _ in cleaned_txt
        ]

        for i, (el, batch_images) in enumerate(zip(messages, images)):
            # optionally add the images per batch element.
            if batch_images is not None:
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image_obj} for image_obj in batch_images],
                    }
                )
            # add the text.
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


# Copied from diffusers.pipelines.flux2.pipeline_flux2._validate_and_process_images
def _validate_and_process_images(
    images: list[list[PIL.Image.Image]] | list[PIL.Image.Image],
    image_processor: Flux2ImageProcessor,
    upsampling_max_image_size: int,
) -> list[list[PIL.Image.Image]]:
    # Simple validation: ensure it's a list of PIL images or list of lists of PIL images
    if not images:
        return []

    # Check if it's a list of lists or a list of images
    if isinstance(images[0], PIL.Image.Image):
        # It's a list of images, convert to list of lists
        images = [[im] for im in images]

    # potentially concatenate multiple images to reduce the size
    images = [[image_processor.concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in images]

    # cap the pixels
    images = [
        [image_processor._resize_if_exceeds_area(img_i, upsampling_max_image_size) for img_i in img_i]
        for img_i in images
    ]
    return images


class Flux2Pipeline(Flux2PipelineBase):
    """Flux2 pipeline for text-to-image generation."""

    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config, prefix=prefix)

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self._execution_device)
        self.tokenizer = PixtralProcessor.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )

        self.system_message = SYSTEM_MESSAGE
        self.system_message_upsampling_t2i = SYSTEM_MESSAGE_UPSAMPLING_T2I
        self.system_message_upsampling_i2i = SYSTEM_MESSAGE_UPSAMPLING_I2I
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

    @staticmethod
    def _get_mistral_3_small_prompt_embeds(
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        prompt: str | list[str],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        system_message: str = SYSTEM_MESSAGE,
        hidden_states_layers: list[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
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
        output = text_encoder(
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

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.upsample_prompt
    def upsample_prompt(
        self,
        prompt: str | list[str],
        images: list[PIL.Image.Image] | list[list[PIL.Image.Image]] = None,
        temperature: float = 0.15,
        device: torch.device = None,
    ) -> list[str]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = self.text_encoder.device if device is None else device

        # Set system message based on whether images are provided
        if images is None or len(images) == 0 or images[0] is None:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I

        # Validate and process the input images
        if images:
            images = _validate_and_process_images(images, self.image_processor, self.upsampling_max_image_size)

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message, images=images)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device, self.text_encoder.dtype)

        # Generate text using the model's generate method
        generated_ids = self.text_encoder.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
        )

        # Decode only the newly generated tokens (skip input tokens)
        # Extract only the generated portion
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        upsampled_prompt = self.tokenizer.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return upsampled_prompt

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (10, 20, 30),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_mistral_3_small_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
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

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in ["latents", "prompt_embeds"] for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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

    def forward(
        self,
        req: OmniDiffusionRequest,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (10, 20, 30),
        caption_upsample_temperature: float = None,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        if (
            raw_image := None
            if isinstance(first_prompt, str)
            else first_prompt.get("multi_modal_data", {}).get("image")
        ) is None:
            pass  # use image from param list
        elif isinstance(raw_image, list):
            image = [PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image, im) for im in raw_image]
        else:
            image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else cast(PIL.Image.Image, raw_image)

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        text_encoder_out_layers = req.sampling_params.extra_args.get("text_encoder_out_layers", text_encoder_out_layers)

        req_prompt_embeds = [p.get("prompt_embeds") if not isinstance(p, str) else None for p in req.prompts]
        if any(p is not None for p in req_prompt_embeds):
            # If at list one prompt is provided as an embedding,
            # Then assume that the user wants to provide embeddings for all prompts, and enter this if block
            # If the user in fact provides mixed input format, req_prompt_embeds will have some None's
            # And `torch.stack` automatically raises an exception for us
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore # intentionally expect TypeError

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. prepare text embeddings
        if caption_upsample_temperature:
            prompt = self.upsample_prompt(prompt, images=image, temperature=caption_upsample_temperature, device=device)
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
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

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
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

        # 7. Denoising loop
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

            noise_pred = self.transformer(
                hidden_states=latent_model_input,  # (B, image_seq_len, C)
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,  # B, text_seq_len, 4
                img_ids=latent_image_ids,  # B, image_seq_len, 4
                joint_attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

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

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
