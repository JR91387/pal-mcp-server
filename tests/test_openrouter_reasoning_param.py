"""OpenRouter `reasoning` request field and reasoning capture on the chat-completions path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from providers.openai_compatible import OpenAICompatibleProvider
from providers.shared import ModelCapabilities, ProviderType


def _caps(thinking=True, default_effort=None):
    return ModelCapabilities(
        provider=ProviderType.OPENROUTER,
        model_name="vendor/model",
        friendly_name="test",
        context_window=100000,
        max_output_tokens=4096,
        supports_extended_thinking=thinking,
        default_reasoning_effort=default_effort,
    )


def _make(cls_flag, caps):
    class P(OpenAICompatibleProvider):
        FRIENDLY_NAME = "Test"
        SUPPORTS_REASONING_PARAM = cls_flag

        def get_provider_type(self):
            return ProviderType.OPENROUTER if cls_flag else ProviderType.CUSTOM

        def get_capabilities(self, model_name):
            return caps

        def validate_model_name(self, model_name):
            return True

        def list_models(self, **kwargs):
            return ["vendor/model"]

    with patch("providers.openai_compatible.OpenAI"):
        p = P("key", base_url="http://localhost")
    p._client = MagicMock()
    return p


def _response(message, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="vendor/model",
        id="r1",
        created=1,
        usage=usage,
    )


def _run(provider, response, **kwargs):
    create = provider.client.chat.completions.create
    create.return_value = response
    result = provider.generate_content(prompt="hi", model_name="vendor/model", **kwargs)
    return result, create.call_args.kwargs


@pytest.mark.parametrize(
    "mode,expected",
    [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "high")],
)
def test_effort_mapping(mode, expected):
    p = _make(True, _caps())
    _, params = _run(p, _response(SimpleNamespace(content="ok")), thinking_mode=mode)
    assert params["extra_body"] == {"reasoning": {"effort": expected}}


def test_default_reasoning_effort_used_without_thinking_mode():
    p = _make(True, _caps(default_effort="high"))
    _, params = _run(p, _response(SimpleNamespace(content="ok")))
    assert params["extra_body"] == {"reasoning": {"effort": "high"}}


def test_nothing_sent_without_mode_or_default():
    p = _make(True, _caps())
    _, params = _run(p, _response(SimpleNamespace(content="ok")))
    assert "extra_body" not in params


def test_absent_for_non_reasoning_model():
    p = _make(True, _caps(thinking=False, default_effort="high"))
    _, params = _run(p, _response(SimpleNamespace(content="ok")), thinking_mode="high")
    assert "extra_body" not in params


def test_absent_for_non_openrouter_provider():
    p = _make(False, _caps(default_effort="high"))
    _, params = _run(p, _response(SimpleNamespace(content="ok")), thinking_mode="high")
    assert "extra_body" not in params


def test_reasoning_captured_in_metadata_not_content():
    details = [{"type": "reasoning.text", "text": "step"}]
    msg = SimpleNamespace(content="answer", reasoning="thinking...", reasoning_details=details)
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=5,
        total_tokens=6,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    p = _make(True, _caps())
    result, _ = _run(p, _response(msg, usage), thinking_mode="medium")
    assert result.content == "answer"
    assert "thinking..." not in result.content
    assert result.metadata["reasoning"] == "thinking..."
    assert result.metadata["reasoning_details"] == details
    assert result.metadata["reasoning_tokens"] == 3


def test_no_reasoning_metadata_when_absent():
    p = _make(True, _caps())
    result, _ = _run(p, _response(SimpleNamespace(content="answer")), thinking_mode="medium")
    assert "reasoning" not in result.metadata
    assert "reasoning_details" not in result.metadata
