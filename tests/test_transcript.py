"""Tests for transcript_extraction module."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from components.transcript_extraction import transcribe_audio, WhisperModelSingleton


@pytest.fixture(autouse=True)
def reset_singleton():
    WhisperModelSingleton._instance = None
    yield
    WhisperModelSingleton._instance = None


@pytest.mark.asyncio
async def test_transcribe_audio():
    with patch("components.transcript_extraction.WhisperModelSingleton") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.transcribe = MagicMock(return_value="hello world")
        mock_cls.get_instance.return_value = mock_instance
        result = await transcribe_audio("/path/to/audio.wav")
        assert result == "hello world"
        mock_instance.transcribe.assert_called_once_with("/path/to/audio.wav")
