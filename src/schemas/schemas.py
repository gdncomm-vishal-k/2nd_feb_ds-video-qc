from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from configs import config
import re

# Request Schema
class VideoQCRequest(BaseModel):
    request_id: str
    sku_id: List[str]
    caption: str = ""
    video_id: str
    video_path: str

    @validator("request_id")
    def validate_request_id(cls, value):
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("request_id cannot be empty or only whitespace")
        return stripped_value

    @validator("sku_id")
    def validate_sku_id_count(cls, sku_id):
        if len(sku_id) > config.MAX_SKU_ID_COUNT:
            raise ValueError(f"sku_id must not contain more than {config.MAX_SKU_ID_COUNT} items")
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
        if len(video_id) > config.MAX_VIDEO_ID_LENGTH:
            raise ValueError(f"video_id must not exceed {config.MAX_VIDEO_ID_LENGTH} characters")
        return video_id

    @validator("video_path")
    def validate_video_path_constraints(cls, video_path):
        if len(video_path) > config.MAX_VIDEO_PATH_LENGTH:
            raise ValueError(f"video_path must not exceed {config.MAX_VIDEO_PATH_LENGTH} characters")
        if not video_path.startswith(("http://", "https://")):
            raise ValueError("video_path must start with http:// or https://")
        return video_path


# Response Schemas
class AudioQC(BaseModel):
    rejected: bool
    rejected_words: List[str] = []


class CaptionQC(BaseModel):
    rejected: bool
    rejected_words: List[str] = []


class VideoQCResponse(BaseModel):
    success: bool = True
    request_id: str
    sku_id: List[str]
    caption: str
    video_id: str
    video_path: str
    video_qc: Dict[str, List[str]]
    audio_qc: AudioQC
    caption_qc: CaptionQC


class VideoQCErrorResponse(BaseModel):
    success: bool = False
    request_id: str
    sku_id: List[str]
    caption: str
    video_id: str
    video_path: str
    error: str
