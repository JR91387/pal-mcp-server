"""Diagnostics for model responses that came back with no content."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_empty_response_diagnostics(model_response: Any, model_name: str) -> dict[str, Any]:
    """Explain why a model response was empty, using provider metadata."""
    metadata = getattr(model_response, "metadata", None) or {}
    usage = getattr(model_response, "usage", None) or {}
    finish_reason = metadata.get("finish_reason")
    reasoning_tokens = metadata.get("reasoning_tokens")
    generation_id = metadata.get("generation_id") or metadata.get("id")
    output_tokens = usage.get("output_tokens")

    if str(finish_reason).lower() == "length":
        explanation = (
            "Model used its whole output budget on reasoning "
            f"(finish_reason=length, {reasoning_tokens if reasoning_tokens is not None else 'unknown'} reasoning tokens) "
            "and returned no answer; raise max_output_tokens or lower thinking_mode."
        )
    else:
        gen = f" (generation id {generation_id})" if generation_id else ""
        retried = "; retried once" if metadata.get("empty_retried") else ""
        explanation = f"Provider returned an empty body{gen}{retried}."

    return {
        "model": model_name,
        "finish_reason": finish_reason,
        "reasoning_tokens": reasoning_tokens,
        "generation_id": generation_id,
        "output_tokens": output_tokens,
        "explanation": explanation,
    }


def empty_response_result(model_response: Any, model_name: str) -> dict[str, Any]:
    """Build the workflow-tool result for an empty response and log it at WARNING."""
    diagnostics = build_empty_response_diagnostics(model_response, model_name)
    logger.warning("Empty model response: %s | diagnostics=%s", diagnostics["explanation"], diagnostics)
    return {
        "error": f"No response from model: {diagnostics['explanation']}",
        "status": "empty_response",
        "diagnostics": diagnostics,
    }
