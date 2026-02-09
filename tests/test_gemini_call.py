"""Tests for gemini_call module."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from components.gemini_call import get_prompt_ready, validate_text_with_gemini


def test_get_prompt_ready():
    out = get_prompt_ready(prompt="System prompt", input_text="user input")
    assert "System prompt" in out
    assert "user input" in out
    assert "predict for the input" in out


@pytest.mark.asyncio
async def test_validate_text_with_gemini_empty_input():
    result = await validate_text_with_gemini(input_text="")
    assert result == "NO INPUT TEXT"


@pytest.mark.asyncio
async def test_validate_text_with_gemini_whitespace_only():
    result = await validate_text_with_gemini(input_text="   ")
    assert result == "NO INPUT TEXT"


@pytest.mark.asyncio
async def test_validate_text_with_gemini_none_input():
    result = await validate_text_with_gemini(input_text=None)
    assert result == "NO INPUT TEXT"


@pytest.mark.asyncio
async def test_validate_text_with_gemini_success():
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = '{"FLAG": "False"}'
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    with patch("components.gemini_call.GenerativeModel", return_value=mock_model):
        result = await validate_text_with_gemini(input_text="Hello")
        assert "False" in result


@pytest.mark.asyncio
async def test_validate_text_with_gemini_no_response():
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    with patch("components.gemini_call.GenerativeModel", return_value=mock_model):
        result = await validate_text_with_gemini(input_text="Hello")
        assert result == "NO RESPONSE FROM GEMINI"


@pytest.mark.asyncio
async def test_validate_text_with_gemini_exception():
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("api error"))
    with patch("components.gemini_call.GenerativeModel", return_value=mock_model):
        with pytest.raises(RuntimeError) as exc_info:
            await validate_text_with_gemini(input_text="Hello")
        assert "Text generation failed" in str(exc_info.value)
