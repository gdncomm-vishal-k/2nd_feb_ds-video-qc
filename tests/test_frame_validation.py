"""Tests for frame_X_audio_validation module."""
import pytest
from unittest.mock import AsyncMock, patch

from components.frame_X_audio_validation import (
    process_audio_pipeline,
    validate_frames_with_audio_and_caption,
)


@pytest.mark.asyncio
async def test_process_audio_pipeline():
    with patch("components.frame_X_audio_validation.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
        with patch("components.frame_X_audio_validation.validate_text_with_gemini", new_callable=AsyncMock) as mock_gemini:
            mock_transcribe.return_value = "some transcript"
            mock_gemini.return_value = '{"FLAG": "False", "reason": "ok"}'
            results, flag = await process_audio_pipeline("/path/to/audio.wav")
            assert "False" in results or "ok" in results
            assert flag is False


@pytest.mark.asyncio
async def test_process_audio_pipeline_flag_true():
    with patch("components.frame_X_audio_validation.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
        with patch("components.frame_X_audio_validation.validate_text_with_gemini", new_callable=AsyncMock) as mock_gemini:
            mock_transcribe.return_value = "bad words"
            mock_gemini.return_value = '{"FLAG": "True", "reason": "invalid"}'
            results, flag = await process_audio_pipeline("/path/to/audio.wav")
            assert flag is True


@pytest.mark.asyncio
async def test_validate_frames_with_audio_and_caption_none_urls_raises():
    with pytest.raises(ValueError) as exc_info:
        await validate_frames_with_audio_and_caption(
            frame_urls=None,
            audio_path="/a.wav",
            caption="cap",
            url_mapping={},
        )
    assert "Frame URLs" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_frames_with_audio_and_caption_success():
    with patch("components.frame_X_audio_validation.process_audio_pipeline", new_callable=AsyncMock) as mock_audio:
        with patch("components.frame_X_audio_validation.predict_frames_in_batches", new_callable=AsyncMock) as mock_pred:
            with patch("components.frame_X_audio_validation.validate_text_with_gemini", new_callable=AsyncMock) as mock_gemini:
                mock_audio.return_value = ("audio reason", False)
                mock_pred.return_value = {"blur": [], "nsfw": []}
                mock_gemini.return_value = "False"
                frame_pred, audio_res, audio_flag, caption_flag, caption_res = await validate_frames_with_audio_and_caption(
                    frame_urls=["https://x/frame_0001.jpg"],
                    audio_path="/a.wav",
                    caption="Hi",
                    url_mapping={},
                )
                assert frame_pred == {"blur": [], "nsfw": []}
                assert audio_res == "audio reason"
                assert audio_flag is False
                assert caption_flag is False
                assert caption_res == "False"
