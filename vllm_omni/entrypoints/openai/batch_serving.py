# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time

from fastapi import Request
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionMessageParam,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


class OmniOpenAIServingChatBatch(OmniOpenAIServingChat):
    async def create_batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Given a request, submit each request to chat completions & collect the results."""
        model = ""
        chat_requests: list[ChatCompletionMessageParam] = []
        choices: list[ChatCompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for msg in request.messages:
            chat_cmp_request = request.to_chat_completion_request(msg)
            chat_requests.append(chat_cmp_request)

        # Submit each chat completion request as a task, then gather results.
        # TODO (Alex): This can probably be done more optimally
        tasks = [asyncio.create_task(self.create_chat_completion(c, raw_request)) for c in chat_requests]
        results = await asyncio.gather(*tasks)

        for i, resp in enumerate(results):
            if isinstance(resp, ErrorResponse):
                return resp
            if not isinstance(resp, ChatCompletionResponse):
                return self.create_error_response(f"Unexpected response type for conversation {i}")
            completion: ChatCompletionResponse = resp
            model = completion.model
            for choice in completion.choices:
                choice.index = i
                choices.append(choice)
            if completion.usage:
                total_prompt_tokens += completion.usage.prompt_tokens
                total_completion_tokens += completion.usage.completion_tokens

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-batch-{int(time.time())}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )
