import sys
from pathlib import Path

# Add src/ to path for components imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from components.transcript_extraction import transcribe_audio
from components.gemini_call import validate_text_with_gemini
from components.image_qc_utils.image_qc_predictions import predict_frames_in_batches
from configs.logging import simple_logger

@simple_logger()
async def process_audio_pipeline(audio_path):
    transcript = await transcribe_audio(audio_path)
    gemini_results = await validate_text_with_gemini(input_text=transcript)
    audio_qc_flag = True if 'true' in gemini_results.lower() else False
    return gemini_results, audio_qc_flag


@simple_logger()
async def validate_frames_with_audio_and_caption(frame_urls = None, audio_path = None, caption = None):
    if frame_urls is None:
        raise ValueError("Frame URLs are required")
    

    tasks = [
        process_audio_pipeline(audio_path),
        predict_frames_in_batches(frame_urls),
        validate_text_with_gemini(input_text=caption)
    ]
    results = await asyncio.gather(*tasks)

    audio_results, audio_qc_flag = results[0]

    frame_predictions = results[1]

    caption_results = results[2]
    caption_qc_flag = True if 'true' in caption_results.lower() else False

    return frame_predictions, audio_results, audio_qc_flag, caption_qc_flag, caption_results



