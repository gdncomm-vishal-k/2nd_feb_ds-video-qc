import sys
from pathlib import Path

# Add src/ to path for components and schemas imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import shutil
import asyncio
from schemas.schemas import VideoQCResponse, VideoQCRequest
from components.frame_X_audio_extraction import process_video_pipeline
from components.frame_X_audio_validation import validate_frames_with_audio_and_caption
from configs.config import VIDEO_QC_CHUNKS, VIDEO_QC_FPS, VIDEO_QC_BUCKET, VIDEO_QC_GCS_FOLDER, VIDEO_QC_TEMP_FOLDER, PREDICTIONS
from configs.logging import simple_logger


def _rejected_words_to_list(value):
    """Convert rejected_words string to list: [] for None/'None', else ['word1', 'word2']."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts if parts else []
    return []


@simple_logger()
def delete_video_folder(video_folder: Path):
    try:
        shutil.rmtree(video_folder)
    except Exception as e:
        raise Exception(f"Error deleting video folder: {e}")

@simple_logger()
async def run_video_qc_pipeline(request: VideoQCRequest) -> dict:
    """
    Main pipeline for Video QC processing.
    
    Args:
        request: VideoQCRequest with sku_id, caption, video_id, video_path
    
    Returns:
        VideoQCResponse as dict
    """
    
    # 1. Process video - extract frames and audio
    frame_urls, audio_path, url_mapping = await process_video_pipeline(
        video_url=request.video_path,
        Videos_folder=VIDEO_QC_TEMP_FOLDER,
        chunks=VIDEO_QC_CHUNKS,
        fps=VIDEO_QC_FPS,
        bucket=VIDEO_QC_BUCKET,
        gcs_base_folder=VIDEO_QC_GCS_FOLDER
    )
    
    # 2. Validate frames, audio, and caption
    frame_predictions, audio_results, audio_qc_flag, caption_qc_flag, caption_results = await validate_frames_with_audio_and_caption(
        frame_urls=frame_urls,
        audio_path=audio_path,
        caption=request.caption,
        url_mapping=url_mapping
    )
    
    # frame_predictions is already summarized from predict_frames_in_batches
    
    # 3. Cleanup: delete video folder
    delete_video_folder(Path(audio_path).parent)

    # 4. Build video_qc dynamically from PREDICTIONS config
    video_qc = {pred: frame_predictions.get(pred, []) for pred in PREDICTIONS}
    
    # 5. Build response
    response = {
        "success": True,
        "request_id": request.request_id,
        "sku_id": request.sku_id,
        "caption": request.caption,
        "video_id": request.video_id,
        "video_path": request.video_path,
        "video_qc": video_qc,
        "audio_qc": {
            "rejected": audio_qc_flag,
            "rejected_words": _rejected_words_to_list(audio_results)
        },
        "caption_qc": {
            "rejected": caption_qc_flag,
            "rejected_words": _rejected_words_to_list(caption_results)
        }
    }
    
    # 6. Validate response schema
    validated_response = VideoQCResponse(**response)
    
    return validated_response.model_dump()




