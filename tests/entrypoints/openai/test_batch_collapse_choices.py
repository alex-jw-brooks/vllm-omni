import pytest
from openai.types.chat.chat_completion_audio import ChatCompletionAudio
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatMessage,
)

from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

collapse = OmniOpenAIServingChatBatch._maybe_collapse_choices


def _text_choice(text="hello"):
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text),
    )


def _audio_choice():
    audio = ChatCompletionAudio(id="a1", data="base64audio", expires_at=0, transcript="")
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=None, audio=audio),
    )


def _image_choice():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    msg = ChatMessage.model_construct(role="assistant")
    object.__setattr__(msg, "content", content)
    return ChatCompletionResponseChoice.model_construct(
        index=0,
        message=msg,
    )


def test_single_text_passthrough():
    result = collapse([_text_choice()])
    assert result.message.content == "hello"


def test_single_image_passthrough():
    result = collapse([_image_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"


def test_text_plus_audio():
    result = collapse([_text_choice(), _audio_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_audio_plus_text_order_independent():
    result = collapse([_audio_choice(), _text_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_image_plus_audio():
    result = collapse([_image_choice(), _audio_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"
    assert result.message.audio.data == "base64audio"


def test_two_content_choices_raises():
    with pytest.raises(ValueError, match="Multiple content choices cannot be set"):
        collapse([_text_choice(), _text_choice()])


def test_three_choices_raises():
    with pytest.raises(ValueError, match="got 3"):
        collapse([_text_choice(), _audio_choice(), _text_choice()])
