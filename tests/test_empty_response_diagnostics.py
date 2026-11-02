"""Empty-response diagnostics and single retry for chat-completions providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from providers.openai_compatible import OpenAICompatibleProvider
from providers.shared import ProviderType
from utils.empty_response import empty_response_result


class _Provider(OpenAICompatibleProvider):
    FRIENDLY_NAME = "Test"

    def get_provider_type(self):
        return ProviderType.CUSTOM

    def get_capabilities(self, model_name):
        raise NotImplementedError

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["m"]

    def get_all_model_capabilities(self):
        return {}

    def _resolve_model_name(self, model_name):
        return model_name


def _resp(content, finish, gid="gen-1", reasoning=None):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    return SimpleNamespace(
        id=gid,
        model="m",
        created=1,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=5, completion_tokens=100, total_tokens=105, completion_tokens_details=details
        ),
    )


def _run(responses):
    provider = _Provider(api_key="k", base_url="http://localhost:1")
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    provider._client = client
    with patch("providers.base.time.sleep"):
        out = provider.generate_content(prompt="hi", model_name="m", temperature=0.5)
    return out, client


def test_length_with_reasoning_not_retried():
    out, client = _run([_resp("", "length", reasoning=100)])
    assert client.chat.completions.create.call_count == 1
    assert out.metadata["reasoning_tokens"] == 100
    result = empty_response_result(out, "m")
    assert result["status"] == "empty_response"
    assert "reasoning" in result["diagnostics"]["explanation"]
    assert result["diagnostics"]["finish_reason"] == "length"


def test_empty_stop_retried_once_then_success():
    out, client = _run([_resp("  ", "stop", "gen-a"), _resp("answer", "stop", "gen-b")])
    assert client.chat.completions.create.call_count == 2
    assert out.content == "answer"
    assert out.metadata["generation_id"] == "gen-b"


def test_empty_twice_reports_generation_id():
    out, client = _run([_resp("", "stop", "gen-a"), _resp("", "stop", "gen-b")])
    assert client.chat.completions.create.call_count == 2
    result = empty_response_result(out, "m")
    assert result["status"] == "empty_response"
    assert result["diagnostics"]["generation_id"] == "gen-b"
    assert "retried once" in result["diagnostics"]["explanation"]
