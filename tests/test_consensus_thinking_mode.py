"""thinking_mode is exposed by the consensus tool and forwarded to every seat."""

import asyncio
from unittest.mock import MagicMock, patch

from tools.consensus import ConsensusRequest, ConsensusTool


def test_schema_includes_thinking_mode():
    schema = ConsensusTool().get_input_schema()
    prop = schema["properties"]["thinking_mode"]
    assert prop["enum"] == ["minimal", "low", "medium", "high", "max"]


def _request(step_number, thinking_mode=None):
    return ConsensusRequest(
        step="Should we adopt X?",
        step_number=step_number,
        total_steps=2,
        next_step_required=True,
        findings="f",
        models=[{"model": "m1", "stance": "for"}, {"model": "m2", "stance": "against"}],
        thinking_mode=thinking_mode,
    )


def _consult(tool, request, idx):
    provider = MagicMock()
    provider.get_provider_type.return_value.value = "mock"
    provider.generate_content.return_value = MagicMock(content="verdict")
    ctx = MagicMock()
    ctx.calculate_token_allocation.return_value = MagicMock(file_tokens=0)
    with (
        patch.object(tool, "get_model_provider", return_value=provider),
        patch("utils.model_context.ModelContext", return_value=ctx),
        patch.object(tool, "_prepare_file_content_for_prompt", return_value=("", [])),
        patch.object(tool, "get_validated_temperature", return_value=(0.2, [])),
    ):
        result = asyncio.run(tool._consult_model(tool.models_to_consult[idx], request))
    assert result["status"] == "success", result
    return provider.generate_content.call_args.kwargs["thinking_mode"]


def _tool_after_step1(thinking_mode):
    tool = ConsensusTool()
    req = _request(1, thinking_mode)
    tool.store_initial_issue(req.step)
    tool._session_thinking_mode = req.thinking_mode
    tool.models_to_consult = req.models
    return tool


def test_given_value_reaches_each_seat():
    tool = _tool_after_step1("high")
    # step 2 request does not repeat thinking_mode: session value must persist
    assert _consult(tool, _request(1, "high"), 0) == "high"
    assert _consult(tool, _request(2, None), 1) == "high"


def test_default_is_medium():
    tool = _tool_after_step1(None)
    assert _consult(tool, _request(1), 0) == "medium"
    assert _consult(tool, _request(2), 1) == "medium"
