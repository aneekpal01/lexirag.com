"""Unit tests validating Nebius Token Factory reasoning_content extraction behavior."""

import pytest
from pydantic import BaseModel, ConfigDict

from app.clients.nebius import extract_reasoning_or_content
from app.core.exceptions import NebiusExtractionError


class MockMessageWithReasoning:
    """Simulates an OpenAI ChatCompletionMessage containing reasoning_content."""
    def __init__(self, reasoning_content: str, content: str = ""):
        self.reasoning_content = reasoning_content
        self.content = content


class MockMessageStandard:
    """Simulates a legacy OpenAI response containing standard content."""
    def __init__(self, content: str):
        self.content = content


class MockPydanticV2Message(BaseModel):
    """Simulates Pydantic v2 OpenAI response where reasoning_content is in model_extra."""
    model_config = ConfigDict(extra="allow")
    content: str = ""


def test_extract_reasoning_content_priority():
    """Verify reasoning_content is prioritized over content for Nemotron models."""
    message = MockMessageWithReasoning(
        reasoning_content="Detailed juridical rationale under Section 185.",
        content="Short summary.",
    )
    result = extract_reasoning_or_content(message)
    assert result == "Detailed juridical rationale under Section 185."


def test_extract_reasoning_content_when_content_is_none():
    """Verify extraction succeeds when standard content is None (typical on Nebius)."""
    message = MockMessageWithReasoning(
        reasoning_content="Section 73 of the Indian Contract Act 1872 governs liquidated damages.",
        content="",
    )
    result = extract_reasoning_or_content(message)
    assert "Section 73" in result


def test_extract_from_model_extra():
    """Verify extraction from Pydantic v2 model_extra dictionary."""
    msg = MockPydanticV2Message(content="")
    msg.model_extra["reasoning_content"] = "Extracted from model_extra reasoning trace."
    result = extract_reasoning_or_content(msg)
    assert result == "Extracted from model_extra reasoning trace."


def test_extract_from_dictionary_payload():
    """Verify extraction when choice.message is returned as a plain dictionary."""
    dict_payload = {
        "reasoning_content": "Statutory threshold under Section 135 CSR is net worth Rs 500 Cr.",
        "content": None,
    }
    result = extract_reasoning_or_content(dict_payload)
    assert "Section 135" in result


def test_fallback_to_standard_content():
    """Verify graceful fallback to content if model did not return reasoning_content."""
    message = MockMessageStandard(content="Fallback statutory advice.")
    result = extract_reasoning_or_content(message)
    assert result == "Fallback statutory advice."


def test_extraction_raises_on_empty_payload():
    """Verify NebiusExtractionError is raised when neither field has text."""
    empty_message = MockMessageWithReasoning(reasoning_content="", content="")
    with pytest.raises(NebiusExtractionError) as exc_info:
        extract_reasoning_or_content(empty_message)

    assert "neither 'reasoning_content' nor 'content'" in str(exc_info.value)


def test_extraction_raises_on_none():
    """Verify NebiusExtractionError is raised when message is None."""
    with pytest.raises(NebiusExtractionError):
        extract_reasoning_or_content(None)
