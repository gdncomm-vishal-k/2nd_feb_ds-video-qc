import sys
import json
import logging
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
def get_gemini_input_text(transcript: str, caption: str) -> str:
    return f"Transcript: {transcript}\nCaption: {caption}"

@simple_logger()
def get_llm_response_and_flag(gemini_results: str):
    logging.info(f"Gemini results: {gemini_results}")
    postprocess_gemini_results = gemini_results.replace("`","").replace("json","")
    logging.info(f"Postprocess Gemini results: {postprocess_gemini_results}")
    try:
        postprocess_gemini_results = json.loads(postprocess_gemini_results)
        caption_llm_response = postprocess_gemini_results.get('CAPTION_QC_REASON', gemini_results)
        audio_llm_response = postprocess_gemini_results.get('AUDIO_QC_REASON', gemini_results)
        return caption_llm_response, \
            True if 'CAPTION_HAS_ISSUE' in gemini_results else False, \
            audio_llm_response, \
            True if 'AUDIO_HAS_ISSUE' in gemini_results else False
    except Exception as e:
        logging.error(f"Error parsing Gemini results: {e}")
        return gemini_results, \
            True if 'CAPTION_HAS_ISSUE' in gemini_results else False, \
            gemini_results, \
            True if 'AUDIO_HAS_ISSUE' in gemini_results else False



@simple_logger()
async def process_audio_and_caption_pipeline(audio_path , caption):
    transcript = await transcribe_audio(audio_path)
    gemini_input_text = get_gemini_input_text(transcript ,caption)
    gemini_results = await validate_text_with_gemini(input_text=gemini_input_text)
    caption_llm_response , caption_qc_flag, audio_llm_response , audio_qc_flag = get_llm_response_and_flag(gemini_results)
    return audio_llm_response, audio_qc_flag, caption_llm_response, caption_qc_flag


@simple_logger()
async def validate_frames_with_audio_and_caption(frame_urls=None, audio_path=None, caption=None, url_mapping=None):
    if frame_urls is None:
        raise ValueError("Frame URLs are required")

    tasks = [
        process_audio_and_caption_pipeline(audio_path, caption),
        predict_frames_in_batches(frame_urls, url_mapping),
    ]
    results = await asyncio.gather(*tasks)

    audio_llm_response, audio_qc_flag, caption_llm_response, caption_qc_flag = results[0]
    frame_predictions = results[1]

    return frame_predictions, audio_llm_response, audio_qc_flag, caption_qc_flag, caption_llm_response



