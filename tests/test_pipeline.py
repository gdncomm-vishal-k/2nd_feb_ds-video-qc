"""
Tests for src/components/pipeline.py.
No mocks – uses real pipeline and real sample request from conftest (SAMPLE_VIDEO_QC_REQUEST).
Integration tests are slow (real video download, frames, GCS, models).
"""

import tempfile
from pathlib import Path

import pytest

from conftest import SAMPLE_VIDEO_QC_REQUEST
from schemas.schemas import VideoQCRequest

from components.pipeline import (
    delete_video_folder,
    process_batch_requests_from_kafka,
    run_video_qc_pipeline,
)


# ---------------------------------------------------------------------------
# delete_video_folder
# ---------------------------------------------------------------------------
def test_delete_video_folder_removes_folder():
    """delete_video_folder(path) removes the folder and its contents."""
    tmp = tempfile.mkdtemp()
    path = Path(tmp)
    (path / "dummy.txt").write_text("x")
    assert path.exists()

    delete_video_folder(path)

    assert not path.exists()


def test_delete_video_folder_raises_when_path_missing():
    """delete_video_folder(non_existent_path) raises Exception."""
    path = Path("/nonexistent/folder/that/does/not/exist")

    with pytest.raises(Exception) as exc_info:
        delete_video_folder(path)

    assert "Error deleting video folder" in str(exc_info.value)


# ---------------------------------------------------------------------------
# run_video_qc_pipeline – real pipeline (uses SAMPLE_VIDEO_QC_REQUEST from conftest)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_video_qc_pipeline_returns_response_dict(sample_video_qc_request):
    """run_video_qc_pipeline(VideoQCRequest) returns dict with all response keys."""
    request = VideoQCRequest(**sample_video_qc_request)

    result = await run_video_qc_pipeline(request)

    assert isinstance(result, dict)
    assert result["sku_id"] == sample_video_qc_request["sku_id"]
    assert result["video_id"] == sample_video_qc_request["video_id"]
    assert result["caption"] == sample_video_qc_request["caption"]
    assert result["video_path"] == sample_video_qc_request["video_path"]
    assert "video_qc" in result and isinstance(result["video_qc"], dict)
    assert "audio_qc" in result and "rejected" in result["audio_qc"]
    assert "caption_qc" in result and "rejected" in result["caption_qc"]


# ---------------------------------------------------------------------------
# process_batch_requests_from_kafka – real batch (list of request dicts from conftest)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_batch_requests_from_kafka_returns_list_of_responses():
    """process_batch_requests_from_kafka([request_dict]) returns list of response dicts."""
    requests = [SAMPLE_VIDEO_QC_REQUEST.copy()]

    responses = await process_batch_requests_from_kafka(requests)

    assert isinstance(responses, list)
    assert len(responses) == 1
    r = responses[0]
    assert r["video_id"] == SAMPLE_VIDEO_QC_REQUEST["video_id"]
    assert "video_qc" in r and "audio_qc" in r and "caption_qc" in r
