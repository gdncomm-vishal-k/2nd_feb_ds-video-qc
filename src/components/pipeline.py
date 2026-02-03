import sys
from pathlib import Path

# Add src/ to path for components and schemas imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import shutil
from schemas.schemas import VideoQCRequest, VideoQCResponse
from components.frame_X_audio_extraction import process_video_pipeline
from components.frame_X_audio_validation import validate_frames_with_audio_and_caption
from configs.config import VIDEO_QC_CHUNKS, VIDEO_QC_FPS, VIDEO_QC_BUCKET, VIDEO_QC_GCS_FOLDER, VIDEO_QC_TEMP_FOLDER, PREDICTIONS
from configs.logging import simple_logger

@simple_logger()
def delete_video_folder(video_folder: Path):
    try:
        shutil.rmtree(video_folder)
    except Exception as e:
        raise Exception(f"Error deleting video folder: {e}")

@simple_logger()
async def run_video_qc_pipeline(request: dict) -> dict:
    """
    Main pipeline for Video QC processing.
    
    Args:
        request: Dict with sku_id, caption, video_id, video_path
    
    Returns:
        VideoQCResponse as dict
    """
    # 1. Validate request
    validated_request = VideoQCRequest(**request)
    
    # 2. Process video - extract frames and audio
    frame_urls, audio_path, url_mapping = await process_video_pipeline(
        video_url=validated_request.video_path,
        Videos_folder=VIDEO_QC_TEMP_FOLDER,
        chunks=VIDEO_QC_CHUNKS,
        fps=VIDEO_QC_FPS,
        bucket=VIDEO_QC_BUCKET,
        gcs_base_folder=VIDEO_QC_GCS_FOLDER
    )
    
    # 3. Validate frames, audio, and caption
    frame_predictions, audio_results, audio_qc_flag, caption_qc_flag, caption_results = await validate_frames_with_audio_and_caption(
        frame_urls=frame_urls,
        audio_path=audio_path,
        caption=validated_request.caption,
        url_mapping=url_mapping
    )
    
    # frame_predictions is already summarized from predict_frames_in_batches
    
    # 4. Cleanup: delete video folder
    delete_video_folder(Path(audio_path).parent)

    # 5. Build video_qc dynamically from PREDICTIONS config
    video_qc = {pred: frame_predictions.get(pred, []) for pred in PREDICTIONS}
    
    # 6. Build response
    response = {
        "sku_id": validated_request.sku_id,
        "caption": validated_request.caption,
        "video_id": validated_request.video_id,
        "video_path": validated_request.video_path,
        "video_qc": video_qc,
        "audio_qc": {
            "rejected": audio_qc_flag,
            "rejected_reason": audio_results
        },
        "caption_qc": {
            "rejected": caption_qc_flag,
            "rejected_reason": caption_results
        }
    }
    
    # 7. Validate response schema
    validated_response = VideoQCResponse(**response)
    
    return validated_response.model_dump()
