"""
Tests for src/schemas/schemas.py.
Uses real request/response examples; no mocks. Import schemas, build instances, assert.
"""

import pytest
from pydantic import ValidationError

from src.schemas.schemas import (
    AudioQC,
    CaptionQC,
    VideoQCRequest,
    VideoQCResponse,
)

# ----- Realistic samples (from your examples) -----

VALID_REQUEST = {
    "request_id": "12345678910",
    "sku_id": ["BRO-70057-00002-00001", "BRO-70057-00002-00002"],
    "caption": "bullshit",
    "video_id": "a66e342b-f4bf-471c-8810-e1157a8e677e",
    "video_path": "https://storage.googleapis.com/test-images-image-qc/Video_QC/sample_videos/(FMU)%20Buttonscarves%20Champ%20de%20Fleurs%20Voile%20Square%20-%20Tabebuya_Reels-REVISI.mp4",
}

VALID_RESPONSE = {
    "request_id": "12345678910",
    "sku_id": ["BRO-70057-00002-00001", "BRO-70057-00002-00002"],
    "caption": "bullshit",
    "video_id": "a66e342b-f4bf-471c-8810-e1157a8e677e",
    "video_path": "https://storage.googleapis.com/test-images-image-qc/Video_QC/sample_videos/(FMU)%20Buttonscarves%20Champ%20de%20Fleurs%20Voile%20Square%20-%20Tabebuya_Reels-REVISI.mp4",
    "video_qc": {
        "pharma_banned": [],
        "pharma_prescription": [],
        "competitor_logo": [],
        "nsfw": [],
        "cigarette": [],
        "alcohol": [],
        "guns": [],
        "blur": ["1-18"],
        "text_predictions": [],
        "water_mark": [],
    },
    "audio_qc": {
        "rejected": False,
        "rejected_words": [],
    },
    "caption_qc": {
        "rejected": True,
        "rejected_words": ["bullshit"],
    },
}


# ---------------------------------------------------------------------------
# VideoQCRequest – valid
# ---------------------------------------------------------------------------
def test_video_qc_request_valid():
    req = VideoQCRequest(**VALID_REQUEST)
    assert req.request_id == VALID_REQUEST["request_id"]
    assert req.sku_id == VALID_REQUEST["sku_id"]
    assert req.caption == VALID_REQUEST["caption"]
    assert req.video_id == VALID_REQUEST["video_id"]
    assert req.video_path == VALID_REQUEST["video_path"]


def test_video_qc_request_caption_empty_default():
    data = {**VALID_REQUEST, "caption": ""}
    req = VideoQCRequest(**data)
    assert req.caption == ""


def test_video_qc_request_caption_any_string():
    """Caption only validated as string; digits/symbols allowed (per current schema)."""
    data = {**VALID_REQUEST, "caption": "hello123 and symbols! @#"}
    req = VideoQCRequest(**data)
    assert req.caption == "hello123 and symbols! @#"


# ---------------------------------------------------------------------------
# VideoQCRequest – invalid (validation errors)
# ---------------------------------------------------------------------------
def test_video_qc_request_invalid_sku_id_too_many():
    data = {**VALID_REQUEST, "sku_id": [f"SKU-{i}" for i in range(31)]}
    with pytest.raises(ValidationError):
        VideoQCRequest(**data)


def test_video_qc_request_invalid_video_id_empty():
    data = {**VALID_REQUEST, "video_id": "   "}
    with pytest.raises(ValidationError):
        VideoQCRequest(**data)


def test_video_qc_request_invalid_video_id_too_long():
    data = {**VALID_REQUEST, "video_id": "x" * 257}
    with pytest.raises(ValidationError):
        VideoQCRequest(**data)


def test_video_qc_request_invalid_video_path_not_http():
    data = {**VALID_REQUEST, "video_path": "ftp://example.com/video.mp4"}
    with pytest.raises(ValidationError):
        VideoQCRequest(**data)


def test_video_qc_request_invalid_video_path_too_long():
    data = {**VALID_REQUEST, "video_path": "https://x.co/" + "a" * 4090}
    with pytest.raises(ValidationError):
        VideoQCRequest(**data)


# ---------------------------------------------------------------------------
# AudioQC
# ---------------------------------------------------------------------------
def test_audio_qc_valid():
    audio = AudioQC(rejected=False, rejected_words=["No issues."])
    assert audio.rejected is False
    assert audio.rejected_words == ["No issues."]


def test_audio_qc_rejected_words_default():
    audio = AudioQC(rejected=True)
    assert audio.rejected is True
    assert audio.rejected_words == []


# ---------------------------------------------------------------------------
# CaptionQC
# ---------------------------------------------------------------------------
def test_caption_qc_valid():
    caption = CaptionQC(rejected=True, rejected_words=["Prohibited word."])
    assert caption.rejected is True
    assert caption.rejected_words == ["Prohibited word."]


# ---------------------------------------------------------------------------
# VideoQCResponse – valid (realistic response)
# ---------------------------------------------------------------------------
def test_video_qc_response_valid():
    resp = VideoQCResponse(**VALID_RESPONSE)
    assert resp.request_id == VALID_RESPONSE["request_id"]
    assert resp.sku_id == VALID_RESPONSE["sku_id"]
    assert resp.caption == VALID_RESPONSE["caption"]
    assert resp.video_id == VALID_RESPONSE["video_id"]
    assert resp.video_path == VALID_RESPONSE["video_path"]
    assert resp.video_qc == VALID_RESPONSE["video_qc"]
    assert resp.video_qc["blur"] == ["1-18"]
    assert resp.audio_qc.rejected is False
    assert resp.audio_qc.rejected_words == []
    assert resp.caption_qc.rejected is True
    assert resp.caption_qc.rejected_words == ["bullshit"]
