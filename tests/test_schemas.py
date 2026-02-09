"""Tests for schemas (request/response models)."""
import pytest
from pydantic import ValidationError

from schemas.schemas import (
    VideoQCRequest,
    VideoQCResponse,
    AudioQC,
    CaptionQC,
)


# --- VideoQCRequest ---


def test_video_qc_request_valid():
    req = VideoQCRequest(
        sku_id=["sku1"],
        caption="Hello world",
        video_id="vid123",
        video_path="https://example.com/video.mp4",
    )
    assert req.sku_id == ["sku1"]
    assert req.caption == "Hello world"
    assert req.video_id == "vid123"
    assert req.video_path == "https://example.com/video.mp4"


def test_video_qc_request_caption_empty_allowed():
    req = VideoQCRequest(
        sku_id=["s"],
        caption="",
        video_id="v",
        video_path="http://a.b/c",
    )
    assert req.caption == ""


def test_video_qc_request_sku_id_max_30():
    req = VideoQCRequest(
        sku_id=[f"s{i}" for i in range(30)],
        caption="",
        video_id="v",
        video_path="https://a/b",
    )
    assert len(req.sku_id) == 30


def test_video_qc_request_sku_id_more_than_30_raises():
    with pytest.raises(ValidationError) as exc:
        VideoQCRequest(
            sku_id=[f"s{i}" for i in range(31)],
            caption="",
            video_id="v",
            video_path="https://a/b",
        )
    assert "30" in str(exc.value)


def test_video_qc_request_caption_invalid_chars_raises():
    with pytest.raises(ValidationError):
        VideoQCRequest(
            sku_id=["s"],
            caption="Hello 123",
            video_id="v",
            video_path="https://a/b",
        )


def test_video_qc_request_caption_letters_and_spaces_only():
    req = VideoQCRequest(
        sku_id=["s"],
        caption="Abc Def",
        video_id="v",
        video_path="https://a/b",
    )
    assert req.caption == "Abc Def"


def test_video_qc_request_video_id_empty_raises():
    with pytest.raises(ValidationError):
        VideoQCRequest(
            sku_id=["s"],
            caption="",
            video_id="   ",
            video_path="https://a/b",
        )


def test_video_qc_request_video_id_over_256_raises():
    with pytest.raises(ValidationError):
        VideoQCRequest(
            sku_id=["s"],
            caption="",
            video_id="x" * 257,
            video_path="https://a/b",
        )


def test_video_qc_request_video_path_over_4096_raises():
    with pytest.raises(ValidationError):
        VideoQCRequest(
            sku_id=["s"],
            caption="",
            video_id="v",
            video_path="https://" + "a" * 4090,
        )


def test_video_qc_request_video_path_must_start_http_or_https():
    with pytest.raises(ValidationError):
        VideoQCRequest(
            sku_id=["s"],
            caption="",
            video_id="v",
            video_path="ftp://example.com/v.mp4",
        )


def test_video_qc_request_video_path_http_ok():
    req = VideoQCRequest(
        sku_id=["s"],
        caption="",
        video_id="v",
        video_path="http://example.com/v.mp4",
    )
    assert req.video_path.startswith("http://")


# --- AudioQC / CaptionQC ---


def test_audio_qc():
    a = AudioQC(rejected=True, rejected_reason="bad")
    assert a.rejected is True
    assert a.rejected_reason == "bad"


def test_audio_qc_default_reason():
    a = AudioQC(rejected=False)
    assert a.rejected_reason == ""


def test_caption_qc():
    c = CaptionQC(rejected=False, rejected_reason="")
    assert c.rejected is False


# --- VideoQCResponse ---


def test_video_qc_response_valid():
    resp = VideoQCResponse(
        sku_id=["s1"],
        caption="Cap",
        video_id="v1",
        video_path="https://x/y",
        video_qc={"blur": ["1-3"], "nsfw": []},
        audio_qc=AudioQC(rejected=False, rejected_reason=""),
        caption_qc=CaptionQC(rejected=False, rejected_reason=""),
    )
    assert resp.sku_id == ["s1"]
    assert resp.video_qc["blur"] == ["1-3"]
    assert resp.audio_qc.rejected is False
