"""Tests for gcs_upload module."""
import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from components.gcs_upload import (
    get_public_urls,
    upload_frame,
    upload_frames_to_gcs,
)


def test_get_public_urls():
    bucket = "my-bucket"
    gcs_base_folder = "Video_QC/folder"
    video_name = "vid1"
    frame_paths = [Path("frame_0001.jpg"), Path("frame_0002.jpg")]
    urls = get_public_urls(bucket, gcs_base_folder, video_name, frame_paths)
    assert len(urls) == 2
    assert "storage.googleapis.com" in urls[0]
    assert "my-bucket" in urls[0]
    assert "vid1" in urls[0]
    assert "frame_0001.jpg" in urls[0]


@pytest.mark.asyncio
async def test_upload_frame_success():
    semaphore = asyncio.Semaphore(1)
    resp = MagicMock()
    resp.status = 200
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(return_value=resp)
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post = MagicMock(return_value=post_ctx)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"image data")
        path = Path(f.name)
    try:
        await upload_frame(semaphore, session, path, "bucket", "folder", "video")
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_upload_frame_fails_on_status_300():
    semaphore = asyncio.Semaphore(1)
    resp = MagicMock()
    resp.status = 500
    resp.text = AsyncMock(return_value="error body")
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(return_value=resp)
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post = MagicMock(return_value=post_ctx)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"x")
        path = Path(f.name)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await upload_frame(semaphore, session, path, "bucket", "folder", "video")
        assert "500" in str(exc_info.value) or "Failed" in str(exc_info.value)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_upload_frames_to_gcs():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"x")
        frame_path = Path(f.name)
    try:
        with patch("components.gcs_upload.google.auth.default") as mock_auth:
            with patch("components.gcs_upload.aiohttp.ClientSession") as mock_session:
                creds = MagicMock()
                creds.token = "token"
                mock_auth.return_value = (creds, None)
                resp = MagicMock()
                resp.status = 200
                post_ctx = AsyncMock()
                post_ctx.__aenter__ = AsyncMock(return_value=resp)
                post_ctx.__aexit__ = AsyncMock(return_value=None)
                sess_instance = MagicMock()
                sess_instance.post = MagicMock(return_value=post_ctx)
                session_ctx = AsyncMock()
                session_ctx.__aenter__ = AsyncMock(return_value=sess_instance)
                session_ctx.__aexit__ = AsyncMock(return_value=None)
                mock_session.return_value = session_ctx
                urls = await upload_frames_to_gcs(
                    bucket="b",
                    gcs_base_folder="f",
                    video_name="v",
                    frame_paths=[frame_path],
                )
                assert len(urls) == 1
                assert "storage.googleapis.com" in urls[0]
    finally:
        frame_path.unlink(missing_ok=True)
