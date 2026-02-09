from typing import List, Dict
from pydantic import BaseModel, Field, field_validator
import re

# Request Schema
class VideoQCRequest(BaseModel):
    sku_id: List[str]
    caption: str = ""
    video_id: str
    video_path: str

    @field_validator("sku_id")
    @classmethod
    def validate_sku_id_count(cls, sku_id):
        if len(sku_id) > 30:
            raise ValueError("sku_id must not contain more than 30 items")
        return sku_id

    @field_validator("caption")
    @classmethod
    def validate_caption_text(cls, caption_text):
        if not isinstance(caption_text, str):
            raise ValueError("caption_text must be a string")
        if caption_text and not re.fullmatch(r"[A-Za-z\s]*", caption_text):
            raise ValueError("caption_text must contain only letters and spaces")
        return caption_text

    @field_validator("video_id")
    @classmethod
    def validate_video_id_constraints(cls, video_id):
        if not video_id.strip():
            raise ValueError("video_id must not be empty")
        if len(video_id) > 256:
            raise ValueError("video_id must not exceed 256 characters")
        return video_id

    @field_validator("video_path")
    @classmethod
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
