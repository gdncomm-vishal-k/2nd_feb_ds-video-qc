"""Tests for frame_X_audio_extraction module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from components.frame_X_audio_extraction import (
    get_video_name,
    get_frame_number,
    get_globally_distinct_frames_pixel_only,
    download_video_to_disk,
    extract_frames_and_audio,
    process_video_pipeline,
)


# --- get_video_name ---


def test_get_video_name_simple():
    assert get_video_name("https://example.com/video.mp4") == "video"


def test_get_video_name_with_encoded():
    name = get_video_name("https://example.com/hello%20world.mp4")
    assert " " not in name
    assert "hello" in name and "world" in name


def test_get_video_name_sanitizes_special_chars():
    name = get_video_name("https://example.com/foo@bar#.mp4")
    assert "@" not in name
    assert "#" not in name


# --- get_frame_number ---


def test_get_frame_number():
    assert get_frame_number(Path("frames/frame_0001.jpg")) == 1
    assert get_frame_number(Path("frame_0042.jpg")) == 42


# --- get_globally_distinct_frames_pixel_only (uses PIL/numpy) ---


def test_get_globally_distinct_frames_pixel_only_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        distinct, mapping = get_globally_distinct_frames_pixel_only(str(frames_dir))
        assert distinct == []
        assert mapping == {}


def test_get_globally_distinct_frames_pixel_only_single_frame():
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available")
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        img = Image.new("L", (10, 10), color=128)
        img.save(frames_dir / "frame_0001.jpg")
        distinct, mapping = get_globally_distinct_frames_pixel_only(
            str(frames_dir), diff_threshold=10
        )
        assert len(distinct) == 1
        assert distinct[0].name == "frame_0001.jpg"
        assert mapping[distinct[0]] == [1]


def test_get_globally_distinct_frames_pixel_only_merges_similar():
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available")
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        for i in range(1, 4):
            img = Image.new("L", (10, 10), color=100 + i)
            img.save(frames_dir / f"frame_{i:04d}.jpg")
        distinct, mapping = get_globally_distinct_frames_pixel_only(
            str(frames_dir), diff_threshold=100, resize_to=(128, 128)
        )
        assert len(distinct) >= 1
        assert sum(len(v) for v in mapping.values()) == 3


# --- download_video_to_disk ---


@pytest.mark.asyncio
async def test_download_video_to_disk():
    head_resp = MagicMock()
    head_resp.headers = {"Content-Length": "1000"}
    head_resp.raise_for_status = MagicMock()
    head_ctx = AsyncMock()
    head_ctx.__aenter__ = AsyncMock(return_value=head_resp)
    head_ctx.__aexit__ = AsyncMock(return_value=None)

    get_resp = MagicMock()
    get_resp.raise_for_status = MagicMock()
    get_resp.read = AsyncMock(return_value=b"x" * 500)
    get_ctx = AsyncMock()
    get_ctx.__aenter__ = AsyncMock(return_value=get_resp)
    get_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.head = MagicMock(return_value=head_ctx)
    mock_session.get = MagicMock(return_value=get_ctx)
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("components.frame_X_audio_extraction.aiohttp.ClientSession", return_value=session_ctx):
        with tempfile.TemporaryDirectory() as tmp:
            path, folder_name = await download_video_to_disk(
                "https://example.com/video.mp4", Videos_folder=tmp, chunks=2
            )
            assert Path(path).exists()
            assert Path(path).suffix == ".mp4"
            assert folder_name


# --- extract_frames_and_audio ---


@pytest.mark.asyncio
async def test_extract_frames_and_audio_ffmpeg_fails():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"not a real video")
        video_path = f.name
    try:
        with patch("components.frame_X_audio_extraction.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error"))
            mock_exec.return_value = mock_proc
            with pytest.raises(RuntimeError) as exc_info:
                await extract_frames_and_audio(video_path)
            assert "FFmpeg" in str(exc_info.value)
    finally:
        Path(video_path).unlink(missing_ok=True)


# --- process_video_pipeline ---


@pytest.mark.asyncio
async def test_process_video_pipeline():
    with patch("components.frame_X_audio_extraction.download_video_to_disk", new_callable=AsyncMock) as mock_dl:
        with patch("components.frame_X_audio_extraction.extract_frames_and_audio", new_callable=AsyncMock) as mock_extract:
            with patch("components.frame_X_audio_extraction.get_globally_distinct_frames_pixel_only") as mock_distinct:
                with patch("components.frame_X_audio_extraction.upload_frames_to_gcs", new_callable=AsyncMock) as mock_upload:
                    mock_dl.return_value = ("/tmp/v/v.mp4", "20200101_120000_video")
                    mock_extract.return_value = ("/tmp/v/frames", "/tmp/v/audio.wav")
                    mock_distinct.return_value = (
                        [Path("/tmp/v/frames/frame_0001.jpg")],
                        {Path("/tmp/v/frames/frame_0001.jpg"): [1]},
                    )
                    mock_upload.return_value = ["https://storage.googleapis.com/b/frame_0001.jpg"]
                    urls, audio_path, url_mapping = await process_video_pipeline(
                        "https://example.com/v.mp4",
                        "/tmp",
                        8,
                        1,
                        "bucket",
                        "folder",
                    )
                    assert urls == ["https://storage.googleapis.com/b/frame_0001.jpg"]
                    assert audio_path == "/tmp/v/audio.wav"
                    assert url_mapping["https://storage.googleapis.com/b/frame_0001.jpg"] == [1]
