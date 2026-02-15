import sys
from pathlib import Path

# Add src/ to path for components and schemas imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add root to path for configs imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import aiohttp
import asyncio
from aiokafka.errors import KafkaError
from fastapi import FastAPI, HTTPException
from components import pub_sub
from schemas.schemas import VideoQCRequest, VideoQCResponse
from components.pipeline import run_video_qc_pipeline
from configs.config import (
    TF_SERVING_API_HEALTH_CHECK_URL,
    TORCH_SERVING_API_HEALTH_CHECK_URL,
    CIGARETTE_API_HEALTH_CHECK_URL
)
from configs.logging import simple_logger
from configs.kafka_config import consumer_topics, producer_topic
from components.pub_sub import PubSub
from application import __version__
app = FastAPI(title="ds-video-qc", version=__version__)
global pubsub, task

@app.on_event("startup")
async def on_startup():
    logging.info("started kafka process.")

    global pubsub, task
    try:
        pubsub = pub_sub.PubSub(consumer_topics, producer_topic, process_batch_requests_from_kafka)
        await pubsub.consumer.start()
        await pubsub.producer.start()
        task = asyncio.create_task(pubsub.start_consumer())
        logging.debug("kafka task got created")
    except KafkaError as e:
        logging.error(e)
        await pubsub.consumer.stop()
        await pubsub.producer.stop()
        raise Exception("stopping the server because kafka initialization failed")
    logging.info("successfully started an async task for kafka consumption")

@app.on_event("shutdown")
async def on_app_exit():

    logging.info("shutting down the kafka consumer task")
    global task, pubsub
    if task is not None:
        task.cancel()
    if pubsub is not None:
        await pubsub.consumer.stop()
        await pubsub.producer.stop()

@app.get("/sys-info/health")
def health_check():
    logging.info("inside health check url")
    return {"version": __version__, "status": "UP"}

@simple_logger()
async def process_batch_requests_from_kafka(requests: list[dict]) -> list[dict]:
    responses = []
    tasks = []
    for request in requests:
        tasks.append(asyncio.create_task(predict(request)))
    responses = await asyncio.gather(*tasks)
    return responses

@simple_logger()
async def check_service(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                return resp.status == 200
    except Exception as e:
        raise Exception(f"Error checking service: {e}")

@simple_logger()
@app.get("/sys-info/dependent-services-health-check")
async def dependent_services_health_check():
    """
    Check the health of the dependent services.
    Returns:
        Dictionary with the health of the dependent services.
    """
    tasks = [
        check_service(TF_SERVING_API_HEALTH_CHECK_URL),
        check_service(TORCH_SERVING_API_HEALTH_CHECK_URL),
        check_service(CIGARETTE_API_HEALTH_CHECK_URL)
    ]
    results = await asyncio.gather(*tasks)
    return {
        "tf_serving": results[0],
        "torch_serving" : results[1],
        "cigarette_api": results[2],
    }

@app.post("/predict", response_model=VideoQCResponse)
async def predict(request: VideoQCRequest) -> VideoQCResponse:
    try:
        dependent_services_health = await dependent_services_health_check()
        if dependent_services_health.get('cigarette_api') and dependent_services_health.get('tf_serving') and dependent_services_health.get('torch_serving'):
            logging.info(f"Request: {request.model_dump()}")
            response = await run_video_qc_pipeline(request)
            logging.info(f"Response: {response}")
            return response
        else:
            raise HTTPException(
                500,
                "One or more dependent services are unhealthy. "
                f"Cigarette API: {dependent_services_health.get('cigarette_api')}, "
                f"TF Serving: {dependent_services_health.get('tf_serving')}, "
                f"Torch Serving: {dependent_services_health.get('torch_serving')}"
            )

    except Exception as e:
        logging.error(f"Error in predict: {e}")
        raise HTTPException(500, f"Error in predict: {e}")
