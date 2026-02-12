import sys
from pathlib import Path

# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from threading import Lock
from faster_whisper import WhisperModel
import asyncio
import logging
from configs.logging import simple_logger
from configs.config import DEVICE, WHISPER_MODEL_PATH

class WhisperModelSingleton:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL_PATH,
            device=DEVICE,
            local_files_only=True,
        )

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            language="en"
        )
        return " ".join(seg.text.strip() for seg in segments)


@simple_logger()
async def transcribe_audio(audio_path: str) -> str:
    transcript = await asyncio.to_thread(
        WhisperModelSingleton.get_instance().transcribe, 
        audio_path
    )
    logging.info(f"Transcript: {transcript}")
    return transcript