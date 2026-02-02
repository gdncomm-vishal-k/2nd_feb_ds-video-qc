import sys
from pathlib import Path

# Add root to path for configs imports (go up 3 levels: image_qc_utils -> components -> src -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import httpx
import json

from configs.config import (
    REQUESTS_TIMEOUT,
    TF_SERVING_REST_ADDR,
    TORCH_SERVING_REST_ADDR,
    CIGARETTE_API_REST_ADDR,
)
from configs.logging import simple_logger

HEADERS = {"Content-Type": "application/json"}

@simple_logger()
async def post(url: str, body: dict):
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=REQUESTS_TIMEOUT) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return response

    except (httpx.ConnectError, httpx.TimeoutException):
        raise Exception("Service unavailable or timed out")

    except httpx.HTTPStatusError as e:
        raise Exception(
            f"HTTP {e.response.status_code}: {e.response.text}"
        )

    except json.JSONDecodeError:
        raise Exception("Invalid JSON response")

    except Exception as e:
        raise Exception(str(e))


@simple_logger()
async def tf_serving_post(body: dict):
    return await post(url=TF_SERVING_REST_ADDR, body=body)

@simple_logger()
async def torch_serving_post(body: dict):
    return await post(url=TORCH_SERVING_REST_ADDR, body=body)


@simple_logger()
async def cigarette_api_post(body: dict):
    return await post(url=CIGARETTE_API_REST_ADDR, body=body)
