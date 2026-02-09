"""Tests for pipeline module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from schemas.schemas import VideoQCRequest
from components.pipeline import delete_video_folder, run_video_qc_pipeline


def test_delete_video_folder_success():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "sub"
        folder.mkdir()
        (folder / "file.txt").write_text("x")
        delete_video_folder(folder)
        assert not folder.exists()


def test_delete_video_folder_nonexistent_raises():
    with pytest.raises(Exception) as exc_info:
        delete_video_folder(Path("/nonexistent/path/xyz"))
    assert "Error deleting" in str(exc_info.value) or "delete" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_video_qc_pipeline_success():
    request = VideoQCRequest(
        sku_id=["s1"],
        caption="Hi",
        video_id="v1",
        video_path="https://example.com/v.mp4",
    )
    frame_urls = ["https://storage.googleapis.com/bucket/frame_0001.jpg"]
    url_mapping = {"https://storage.googleapis.com/bucket/frame_0001.jpg": [1, 2, 3]}
    audio_path = "/tmp/some_folder/audio.wav"
    frame_predictions = {
        "pharma_banned": [],
        "pharma_prescription": [],
        "competitor_logo": [],
        "nsfw": [],
        "cigarette": [],
        "alcohol": [],
        "guns": [],
        "blur": [],
        "text_predictions": [],
        "water_mark": [],
    }
    with patch("components.pipeline.process_video_pipeline", new_callable=AsyncMock) as mock_process:
        with patch("components.pipeline.validate_frames_with_audio_and_caption", new_callable=AsyncMock) as mock_validate:
            with patch("components.pipeline.delete_video_folder") as mock_delete:
                mock_process.return_value = (frame_urls, audio_path, url_mapping)
                mock_validate.return_value = (
                    frame_predictions,
                    "reason",
                    False,
                    False,
                    "caption reason",
                )
                result = await run_video_qc_pipeline(request)
                mock_delete.assert_called_once()
                assert result["sku_id"] == ["s1"]
                assert result["video_id"] == "v1"
                assert result["audio_qc"]["rejected"] is False
                assert result["caption_qc"]["rejected"] is False
                assert "video_qc" in result
