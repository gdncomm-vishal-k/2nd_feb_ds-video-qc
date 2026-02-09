# streamlit_app.py
# Run: streamlit run streamlit_app.py

import streamlit as st
import asyncio
import tempfile
import time
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, unquote

import aiohttp
import numpy as np
from PIL import Image
import subprocess
import logging

# ---------------- BASIC CONFIG ----------------
st.set_page_config(layout="wide")
logging.basicConfig(level=logging.INFO)

# ---------------- UTIL FUNCTIONS ----------------
def get_video_name(video_url: str) -> str:
    path = urlparse(video_url).path
    name = Path(path).stem
    name = unquote(name)
    name = re.sub(r"[^\w\- ]+", "", name)
    return name.replace(" ", "_")


async def _download_chunk(session, url, start, end, idx, data):
    headers = {"Range": f"bytes={start}-{end}"}
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        data[idx] = await resp.read()


async def download_video_to_disk(video_url, base_dir, chunks=8):
    video_name = get_video_name(video_url)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    nano = f"{time.time_ns() % 1_000_000_000:09d}"

    folder = Path(base_dir) / f"{timestamp}_{nano}_{video_name}"
    folder.mkdir(parents=True, exist_ok=True)

    video_path = folder / f"{video_name}.mp4"

    async with aiohttp.ClientSession() as session:
        async with session.head(video_url) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers["Content-Length"])

        chunk_size = total_size // chunks
        data = [None] * chunks

        tasks = []
        for i in range(chunks):
            start = i * chunk_size
            end = total_size - 1 if i == chunks - 1 else (i + 1) * chunk_size - 1
            tasks.append(_download_chunk(session, video_url, start, end, i, data))

        await asyncio.gather(*tasks)

    with open(video_path, "wb") as f:
        for chunk in data:
            f.write(chunk)

    return str(video_path)


async def extract_frames(video_path, fps=1):
    video_path = Path(video_path)
    video_dir = video_path.parent

    frames_dir = video_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    frame_pattern = str(frames_dir / "frame_%04d.jpg")
    audio_path = video_dir / "audio.wav"

    # cmd = [
    #     "ffmpeg",
    #     "-y",
    #     "-i", str(video_path),
    #     "-vf", f"fps={fps}",
    #     frame_pattern,
    # ]vf_filter = (
    PADDING_COLOR = "white"
    fps = 1
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
        raise RuntimeError(stderr.decode())

    return str(frames_dir)


def get_frame_number(frame_path: Path) -> int:
    return int(frame_path.stem.split("_")[-1])


def get_globally_distinct_frames_pixel_only(
    frames_dir: str,
    diff_threshold: float,
    resize_to=(128, 128),
):
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    signatures = []
    distinct_frames = []

    for frame_path in frame_files:
        img = (
            Image.open(frame_path)
            .convert("L")
            .resize(resize_to)
        )
        sig = np.asarray(img, dtype=np.float32)

        matched = False
        for prev_sig in signatures:
            if np.mean(np.abs(sig - prev_sig)) <= diff_threshold:
                matched = True
                break

        if not matched:
            signatures.append(sig)
            distinct_frames.append(frame_path)

    return distinct_frames


async def run_pipeline(video_url, thresholds):
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = await download_video_to_disk(video_url, tmpdir)
        frames_dir = await extract_frames(video_path, fps=1)

        for th in thresholds:
            distinct_paths = get_globally_distinct_frames_pixel_only(
                frames_dir,
                diff_threshold=th,
            )
            # Load images into memory before tmpdir is deleted
            results[th] = [Image.open(p).copy() for p in distinct_paths]

    return results

# ---------------- STREAMLIT UI ----------------
st.title("Video Frame Deduplication – Threshold Comparison")

st.sidebar.header("Input")
video_url = st.sidebar.text_input("Video URL")

st.sidebar.header("Thresholds")
t1 = st.sidebar.slider("Column 1", 0, 255, 10)
t2 = st.sidebar.slider("Column 2", 0, 255, 20)
t3 = st.sidebar.slider("Column 3", 0, 255, 30)

run_btn = st.sidebar.button("Process Video")

if video_url.strip():
    st.sidebar.subheader("Video")
    st.sidebar.video(video_url)

col1, col2, col3 = st.columns(3)

if run_btn:
    if not video_url:
        st.error("Please enter a video URL")
    else:
        st.info("Processing video… this may take a minute")

        thresholds = [t1, t2, t3]
        results = asyncio.run(run_pipeline(video_url, thresholds))

        for col, th in zip([col1, col2, col3], thresholds):
            with col:
                st.subheader(f"Threshold = {th}")
                st.caption(f"Distinct frames: {len(results[th])}")

                for img in results[th]:
                    st.image(img, use_container_width=True)
else:
    st.info("Enter a video URL and click **Process Video**")
