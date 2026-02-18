import asyncio
import logging
import aiohttp
from fastapi import FastAPI
from aiokafka.errors import KafkaError
from tenacity import retry, stop_after_attempt, wait_fixed

from ..components import pub_sub
from ..components.pipeline import run_video_qc_pipeline
from ..components.pub_sub import PubSub
from ..schemas.schemas import VideoQCRequest, VideoQCResponse, VideoQCErrorResponse
from configs.config import (
    TF_SERVING_API_HEALTH_CHECK_URL,
    TORCH_SERVING_API_HEALTH_CHECK_URL,
    CIGARETTE_API_HEALTH_CHECK_URL,
    MAX_RETRY_ATTEMPTS,
    MAX_RETRY_WAIT_DEPENDENT_SERVICES
)
from configs.kafka_config import consumer_topics, producer_topic
from configs.logging import simple_logger, request_id_var
from . import __app_name__, __version__

app = FastAPI(title=__app_name__, version=__version__)

pubsub: PubSub | None = None
consumer_task: asyncio.Task | None = None

@app.on_event("startup")
async def startup():
    global pubsub, consumer_task
    logging.info("Starting Kafka consumer...")

    try:
        pubsub = pub_sub.PubSub(
            consumer_topics,
            producer_topic,
            process_batch_requests_from_kafka
        )

        await pubsub.consumer.start()
        await pubsub.producer.start()

        consumer_task = asyncio.create_task(pubsub.start_consumer())
        logging.info("Kafka consumer started successfully.")

    except KafkaError as e:
        logging.error(f"Kafka startup failed: {e}")
        raise RuntimeError("Kafka initialization failed.")


@app.on_event("shutdown")
async def shutdown():
    global pubsub, consumer_task
    logging.info("Shutting down Kafka...")

    if consumer_task:
        consumer_task.cancel()

    if pubsub:
        await pubsub.consumer.stop()
        await pubsub.producer.stop()


@app.get("/sys-info/health")
def health():
    return {"service_name": __app_name__, "version": __version__, "status": "UP"}


async def check_service(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=MAX_RETRY_WAIT_DEPENDENT_SERVICES) as resp:
                return resp.status == 200
    except Exception:
        return False


@app.get("/sys-info/dependent-services-health-check")
async def dependent_services_health_check():
    tf, torch, cigarette = await asyncio.gather(
        check_service(TF_SERVING_API_HEALTH_CHECK_URL),
        check_service(TORCH_SERVING_API_HEALTH_CHECK_URL),
        check_service(CIGARETTE_API_HEALTH_CHECK_URL)
    )

    return {
        "tf_serving": tf,
        "torch_serving": torch,
        "cigarette_api": cigarette,
    }


async def ensure_services_healthy():
    health = await dependent_services_health_check()
    if not all(health.values()):
        raise RuntimeError(f"Dependent services unhealthy: {health}")


@simple_logger()
@retry(stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
       wait=wait_fixed(MAX_RETRY_WAIT))
async def process_batch_requests_from_kafka(requests: list[dict]) -> list[dict]:
    await ensure_services_healthy()
    tasks = [
        predict_for_kafka_consumer(VideoQCRequest(**req))
        for req in requests
    ]
    return await asyncio.gather(*tasks)


@simple_logger()
async def predict_for_kafka_consumer(request: VideoQCRequest):
    request_id_var.set(request.request_id)
    try:
        logging.info(f"Kafka consumer request: {request.model_dump()}")
        response = await run_video_qc_pipeline(request)
        logging.info(f"Kafka consumer response: {response}")
        return VideoQCResponse(**response).model_dump()

    except Exception as e:
        logging.error(f"Kafka prediction error: {e}")
        return VideoQCErrorResponse(
            request_id=request.request_id,
            sku_id=request.sku_id,
            caption=request.caption,
            video_id=request.video_id,
            video_path=request.video_path,
            error=str(e),
        ).model_dump()


@app.post("/predict",
          response_model=VideoQCResponse | VideoQCErrorResponse)
async def predict(request: VideoQCRequest):
    request_id_var.set(request.request_id)
    try:
        await ensure_services_healthy()
        logging.info(f"Request: {request.model_dump()}")
        response = await run_video_qc_pipeline(request)
        logging.info(f"Response: {response}")
        return VideoQCResponse(**response)

    except Exception as e:
        logging.error(f"Predict error: {e}")
        return VideoQCErrorResponse(
            request_id=request.request_id,
            sku_id=request.sku_id,
            caption=request.caption,
            video_id=request.video_id,
            video_path=request.video_path,
            error=str(e),
        )
