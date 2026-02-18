from threading import Lock
from faster_whisper import WhisperModel
import asyncio
import logging
from configs.logging import simple_logger
from configs.config import DEVICE, WHISPER_MODEL_PATH, BEAM_SIZE, VAD_FILTER, LANGUAGE

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
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            language=LANGUAGE,
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