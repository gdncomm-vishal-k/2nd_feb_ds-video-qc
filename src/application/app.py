import sys
from pathlib import Path

# Add src/ to path for components and schemas imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import aiohttp
from fastapi import FastAPI, HTTPException
from schemas.schemas import VideoQCRequest, VideoQCResponse
from components.pipeline import run_video_qc_pipeline
from configs.config import (
    TF_SERVING_API_HEALTH_CHECK_URL,
    TORCH_SERVING_API_HEALTH_CHECK_URL,
    CIGARETTE_API_HEALTH_CHECK_URL
)

app = FastAPI(
    title="Video QC API",
    description="API for Video Quality Control processing",
    version="1.0.0"
)

async def check_service(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                return resp.status == 200
    except Exception as e:
        raise Exception(f"Error checking service: {e}")

@app.get("/health")
async def health_check():
    tf_serving = await check_service(TF_SERVING_API_HEALTH_CHECK_URL)
    torch_serving = await check_service(TORCH_SERVING_API_HEALTH_CHECK_URL)
    cigarette_api = await check_service(CIGARETTE_API_HEALTH_CHECK_URL)
    
    return cigarette_api, tf_serving, torch_serving

@app.post("/predict", response_model=VideoQCResponse)
async def predict(request: VideoQCRequest):
    try:
        cig_api_healthy, tf_serving_healthy, torch_serving_healthy = await health_check()
        if cig_api_healthy and tf_serving_healthy and torch_serving_healthy:
            logging.info(f"Request: {request.model_dump()}")
            response = await run_video_qc_pipeline(request.model_dump())
            logging.info(f"Response: {response}")
            return response
        else:
            raise HTTPException(
                500,
                "One or more dependent services are unhealthy. "
                f"Cigarette API: {cig_api_healthy}, "
                f"TF Serving: {tf_serving_healthy}, "
                f"Torch Serving: {torch_serving_healthy}"
            )

    except Exception as e:
        logging.error(f"Error in predict: {e}")
        raise HTTPException(500, f"Error in predict: {e}")
