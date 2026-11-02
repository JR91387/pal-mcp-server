"""Tests for optional streaming (PAL_STREAM_RESPONSES) and max_retries=0 in OpenAICompatibleProvider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from providers.openai_compatible import OpenAICompatibleProvider


class _Provider(OpenAICompatibleProvider):
    FRIENDLY_NAME = "Test"

    def get_capabilities(self, model_name):
        from providers.shared import ModelCapabilities, ProviderType

        return ModelCapabilities(
            provider=ProviderType.CUSTOM,
            model_name=model_name,
            friendly_name="Test",
            context_window=100000,
            max_output_tokens=1000,
        )

    def get_provider_type(self):
        from providers.shared import ProviderType

        return ProviderType.CUSTOM

    def validate_model_name(self, model_name):
        return True

    def get_preferred_model(self, category, allowed_models):
        return None

    def _resolve_model_name(self, model_name):
        return model_name

    def _ensure_model_allowed(self, *a, **k):
        return None


def _full_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello world"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        model="m-1",
        id="id-1",
        created=123,
    )


def _chunks():
    def ch(text=None, finish=None, usage=None, choices=True):
        c = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)] if choices else [],
            model="m-1",
            id="id-1",
            created=123,
            usage=usage,
        )
        return c

    return [
        ch("Hello"),
        ch(" world", finish="stop"),
        ch(choices=False, usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)),
    ]


def _run(monkeypatch, env, create_result):
    if env is None:
        monkeypatch.delenv("PAL_STREAM_RESPONSES", raising=False)
    else:
        monkeypatch.setenv("PAL_STREAM_RESPONSES", env)
    provider = _Provider(api_key="k", base_url="https://example.invalid/v1")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = create_result
    provider._client = mock_client
    resp = provider.generate_content(prompt="hi", model_name="m-1", temperature=0.5)
    return resp, mock_client.chat.completions.create.call_args.kwargs


def test_default_off_sends_stream_false(monkeypatch):
    _, kwargs = _run(monkeypatch, None, _full_response())
    assert kwargs["stream"] is False
    assert "stream_options" not in kwargs


@pytest.mark.parametrize("val", ["true", "1", "yes", "on"])
def test_env_on_sends_stream_true(monkeypatch, val):
    _, kwargs = _run(monkeypatch, val, iter(_chunks()))
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_stream_assembles_same_response_as_non_stream(monkeypatch):
    plain, _ = _run(monkeypatch, None, _full_response())
    streamed, _ = _run(monkeypatch, "true", iter(_chunks()))
    assert streamed.content == plain.content == "Hello world"
    assert streamed.usage == plain.usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert streamed.model_name == plain.model_name
    assert streamed.metadata == plain.metadata
    assert streamed.metadata["finish_reason"] == "stop"


def test_client_created_with_max_retries_zero(monkeypatch):
    provider = _Provider(api_key="k", base_url="https://example.invalid/v1")
    with patch("providers.openai_compatible.OpenAI") as mock_openai:
        _ = provider.client
    assert mock_openai.call_args.kwargs["max_retries"] == 0


def test_minimal_fallback_client_max_retries_zero(monkeypatch):
    provider = _Provider(api_key="k", base_url="https://example.invalid/v1")
    calls = []

    def fake_openai(**kw):
        calls.append(kw)
        if "http_client" in kw:
            raise RuntimeError("boom")
        return MagicMock()

    with patch("providers.openai_compatible.OpenAI", side_effect=fake_openai):
        _ = provider.client
    assert len(calls) == 2 and calls[1]["max_retries"] == 0
