import sys
from pathlib import Path

# Add src/ to path for components and schemas imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from fastapi import FastAPI, HTTPException
from schemas.schemas import VideoQCRequest, VideoQCResponse
from components.pipeline import run_video_qc_pipeline

app = FastAPI(
    title="Video QC API",
    description="API for Video Quality Control processing",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=VideoQCResponse)
async def predict(request: VideoQCRequest):
    try:
        logging.info(f"Request: {request.model_dump()}")
        response = await run_video_qc_pipeline(request.model_dump())
        logging.info(f"Response: {response}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
