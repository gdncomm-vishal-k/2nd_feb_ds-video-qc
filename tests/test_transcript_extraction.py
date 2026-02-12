"""
Tests for src/components/transcript_extraction.py.
No mocks – uses real audio file and real Whisper model. Slow; run with: pytest tests/test_transcript_extraction.py -v
"""

import os

import pytest

from components.transcript_extraction import (
    WhisperModelSingleton,
    transcribe_audio,
)

# Real audio file path (must exist where tests run)
SAMPLE_AUDIO_PATH = "/Users/vishalkumarhk/Desktop/QB_service/video_qc_main_repo/temp_videos/20260202_053553_996260000_FMU_Buttonscarves_Champ_de_Fleurs_Voile_Square_-_Tabebuya_Reels-REVISI/audio.wav"

# Skip all tests in this file if the sample audio is not present (e.g. on CI)
pytestmark = pytest.mark.skipif(
    not os.path.isfile(SAMPLE_AUDIO_PATH),
    reason=f"Sample audio not found: {SAMPLE_AUDIO_PATH}",
)


# ---------------------------------------------------------------------------
# WhisperModelSingleton – get_instance returns same instance
# ---------------------------------------------------------------------------
def test_singleton_returns_same_instance():
    """get_instance() twice returns the same object."""
    a = WhisperModelSingleton.get_instance()
    b = WhisperModelSingleton.get_instance()
    assert a is b


# ---------------------------------------------------------------------------
# transcribe() – real transcription from sample audio
# ---------------------------------------------------------------------------
def test_transcribe_returns_string():
    """transcribe(audio_path) returns a non-empty string."""
    instance = WhisperModelSingleton.get_instance()
    result = instance.transcribe(SAMPLE_AUDIO_PATH)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# transcribe_audio() – real async transcription
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transcribe_audio_returns_transcript():
    """transcribe_audio(audio_path) returns transcript string."""
    result = await transcribe_audio(SAMPLE_AUDIO_PATH)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
