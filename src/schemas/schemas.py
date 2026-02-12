from typing import List, Dict
from pydantic import BaseModel, Field, validator
import re

# Request Schema
class VideoQCRequest(BaseModel):
    sku_id: List[str]
    caption: str = ""
    video_id: str
    video_path: str

    @validator("sku_id")
    def validate_sku_id_count(cls, sku_id):
        if len(sku_id) > 30:
            raise ValueError("sku_id must not contain more than 30 items")
        return sku_id

    @validator("caption")
    def validate_caption_text(cls, caption_text):
        if not isinstance(caption_text, str):
            raise ValueError("caption_text must be a string")
        return caption_text

    @validator("video_id")
    def validate_video_id_constraints(cls, video_id):
        if not video_id.strip():
            raise ValueError("video_id must not be empty")
        if len(video_id) > 256:
            raise ValueError("video_id must not exceed 256 characters")
        return video_id

    @validator("video_path")
    def validate_video_path_constraints(cls, video_path):
        if len(video_path) > 4096:
            raise ValueError("video_path must not exceed 4096 characters")
        if not video_path.startswith(("http://", "https://")):
            raise ValueError("video_path must start with http:// or https://")
        return video_path


# Response Schemas
class AudioQC(BaseModel):
    rejected: bool
    rejected_reason: str = ""


class CaptionQC(BaseModel):
    rejected: bool
    rejected_reason: str = ""


class VideoQCResponse(BaseModel):
    sku_id: List[str]
    caption: str
    video_id: str
    video_path: str
    video_qc: Dict[str, List[str]]
    audio_qc: AudioQC
    caption_qc: CaptionQC
