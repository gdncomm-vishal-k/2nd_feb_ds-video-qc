import sys
from pathlib import Path

# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import aiohttp
import google.auth
from google.auth.transport.requests import Request
from configs.logging import simple_logger

@simple_logger()
async def upload_frame(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    file_path: Path,
    bucket: str,
    gcs_base_folder: str,
    video_name: str,
):
    async with semaphore:
        object_name = (
            f"{gcs_base_folder.rstrip('/')}/"
            f"{video_name}/"
            f"{file_path.name}"
        )

        url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        params = {
            "uploadType": "media",
            "name": object_name,
        }

        data = file_path.read_bytes()

        async with session.post(url, params=params, data=data) as resp:
            if resp.status >= 300:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed {file_path.name}: {resp.status} {text}"
                )

@simple_logger()
def get_public_urls(bucket: str, gcs_base_folder: str, video_name: str, frame_paths: list[Path]) -> list[str]:
    GCS_PUBLIC_URL_PREFIX = "https://storage.googleapis.com"
    
    return [
        f"{GCS_PUBLIC_URL_PREFIX}/{bucket}/{gcs_base_folder}/{video_name}/{p.name}"
        for p in frame_paths
    ]

@simple_logger()
async def upload_frames_to_gcs(
    bucket: str,
    gcs_base_folder: str,
    video_name: str,
    frame_paths: list[Path],
    max_concurrency: int = 32,
    content_type: str = "image/jpeg",
):
    # ---------- Auth ----------
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.full_control"]
    )
    creds.refresh(Request())

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": content_type,
    }

    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession(
        headers=headers,
        timeout=timeout,
        connector=connector,
    ) as session:
        tasks = [
            upload_frame(
                semaphore=semaphore,
                session=session,
                file_path=frame,
                bucket=bucket,
                gcs_base_folder=gcs_base_folder,
                video_name=video_name,
            )
            for frame in frame_paths
        ]
        await asyncio.gather(*tasks)

    public_urls = get_public_urls(bucket, gcs_base_folder, video_name, frame_paths)
    return public_urls