from typing import List, Dict
from pydantic import BaseModel


# Request Schema
class VideoQCRequest(BaseModel):
    sku_id: List[str]
    caption: str = ""
    video_id: str
    video_path: str


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
