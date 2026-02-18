from pathlib import Path
import re
import time
import asyncio
import aiohttp
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from urllib.parse import urlparse, unquote
from .gcs_upload import upload_frames_to_gcs
from configs.config import FRAME_DIFF_THRESHOLD, FRAME_RESIZE_TO
from google.auth.transport.requests import Request

from configs.logging import simple_logger

def get_video_name(video_url: str) -> str:
    """Extract and sanitize video name from URL."""
    path = urlparse(video_url).path
    name = Path(path).stem
    name = unquote(name)
    name = re.sub(r"[^\w\- ]+", "", name)
    name = name.replace(" ", "_")
    return name

@simple_logger()    
async def _download_chunk(
    session: aiohttp.ClientSession,
    url: str,
    start: int,
    end: int,
    idx: int,
    data: list,
):
    headers = {"Range": f"bytes={start}-{end}"}
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        data[idx] = await resp.read()


@simple_logger()
async def download_video_to_disk(
    video_url: str,
    videos_folder: str,
    chunks: int = 8,
) -> str:
    """
    Downloads the video from video_url into videos_folder.

    Folder structure:
    videos_folder/YYYYMMDD_HHMMSS_<video_name>/<video_name>.mp4

    Returns:
        Absolute path of the saved video file
    """
    
    video_name = get_video_name(video_url)

    # ---- timestamp (seconds + nanoseconds)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    nano = f"{time.time_ns() % 1_000_000_000:09d}"

    folder_name = f"{timestamp}_{nano}_{video_name}"

    base_dir = Path(videos_folder).expanduser()
    video_dir = base_dir / folder_name
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{video_name}.mp4"

    # ---- async download
    async with aiohttp.ClientSession() as session:
        # HEAD request to get size
        async with session.head(video_url) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers["Content-Length"])

        chunk_size = total_size // chunks
        data = [None] * chunks

        tasks = []
        for i in range(chunks):
            start = i * chunk_size
            end = (
                (i + 1) * chunk_size - 1
                if i < chunks - 1
                else total_size - 1
            )
            tasks.append(
                _download_chunk(
                    session,
                    video_url,
                    start,
                    end,
                    i,
                    data,
                )
            )

        await asyncio.gather(*tasks)

    # ---- write file in correct order
    with open(video_path, "wb") as f:
        for chunk in data:
            f.write(chunk)

    return str(video_path), folder_name

@simple_logger()
async def extract_frames_and_audio(
    video_path: str,
    fps: int = 1,
    PADDING_COLOR: str = "white" ,
) -> tuple[str, str]:
    """
    Extract frames and audio from a local video file.

    Output structure:
    <video_folder>/
      ├── frames/frame_0001.jpg
      └── audio.wav

    Returns:
        (frames_dir_path, audio_file_path)
    """

    video_path = Path(video_path).expanduser()
    video_dir = video_path.parent

    frames_dir = video_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    audio_path = video_dir / "audio.wav"

    frame_pattern = str(frames_dir / "frame_%04d.jpg")

    vf_filter = (
        f"fps={fps},"
        f"pad=max(iw\\,ih):max(iw\\,ih):(ow-iw)/2:(oh-ih)/2:{PADDING_COLOR}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),

        # ----- video frames
        "-vf", vf_filter,
        frame_pattern,

        # ----- audio
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{stderr.decode()}")

    return str(frames_dir), str(audio_path)

def get_frame_number(frame_path: Path) -> int:
    return int(frame_path.stem.split("_")[-1])

@simple_logger()
def get_globally_distinct_frames_pixel_only(frames_dir: str, diff_threshold: float = 10, resize_to=(128, 128)):
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    logging.info(f"Total number of frames extracted: {len(frame_files)}")

    signatures = []
    distinct_frames = []
    frame_mapping = {} 

    for frame_path in frame_files:
        frame_num = get_frame_number(frame_path)
        
        img = (
            Image.open(frame_path)
            .convert("L")
            .resize(resize_to)
        )
        sig = np.asarray(img, dtype=np.float32)

        matched_idx = None
        for idx, prev_sig in enumerate(signatures):
            if np.mean(np.abs(sig - prev_sig)) <= diff_threshold:
                matched_idx = idx
                break

        if matched_idx is not None:
            frame_mapping[distinct_frames[matched_idx]].append(frame_num)
        else:
            signatures.append(sig)
            distinct_frames.append(frame_path)
            frame_mapping[frame_path] = [frame_num]

    logging.info(f"Total number of distinct frames: {len(distinct_frames)}")
    logging.info(f"Frame mapping: {frame_mapping}")
    return distinct_frames, frame_mapping

@simple_logger()
async def process_video_pipeline(video_url, videos_folder, chunks, fps, bucket, gcs_base_folder):
    logging.info(f"Processing video pipeline for video URL: {video_url}")
    video_path, video_name = await download_video_to_disk(
        video_url=video_url,
        videos_folder=videos_folder,
        chunks=chunks
    )

    frames_dir, audio_path = await extract_frames_and_audio(
        video_path=video_path,
        fps=fps
    )

    distinct_frames, frame_mapping = get_globally_distinct_frames_pixel_only(
        frames_dir=frames_dir,
        diff_threshold=FRAME_DIFF_THRESHOLD,
        resize_to=FRAME_RESIZE_TO
    )

    urls = await upload_frames_to_gcs(      
        bucket=bucket,
        gcs_base_folder=gcs_base_folder,
        video_name=video_name,
        frame_paths=distinct_frames,
    )
    
    url_mapping = {url: frame_mapping[path] for url, path in zip(urls, distinct_frames)}

    return urls, audio_path, url_mapping