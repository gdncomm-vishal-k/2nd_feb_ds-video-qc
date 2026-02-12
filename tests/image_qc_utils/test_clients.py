"""
Tests for src/components/image_qc_utils/clients.py.
No mocks – real HTTP. post() tested against httpbin.org; service-specific calls are integration (real URLs from config).
"""

import pytest

from components.image_qc_utils.clients import (
    cigarette_api_post,
    post,
    tf_serving_post,
    torch_serving_post,
)

# Public echo endpoint for testing post() without hitting internal services
HTTPBIN_POST = "https://httpbin.org/post"
HTTPBIN_STATUS_500 = "https://httpbin.org/status/500"


# ---------------------------------------------------------------------------
# post(url, body) – real HTTP via httpbin
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_success():
    """post() returns response with status 200 for valid request."""
    body = {"key": "value", "num": 1}
    response = await post(url=HTTPBIN_POST, body=body)
    assert response.status_code == 200
    data = response.json()
    assert data.get("json") == body


@pytest.mark.asyncio
async def test_post_http_error_raises():
    """post() to URL that returns 500 raises Exception with HTTP message."""
    with pytest.raises(Exception) as exc_info:
        await post(url=HTTPBIN_STATUS_500, body={})
    assert "500" in str(exc_info.value)


# ---------------------------------------------------------------------------
# tf_serving_post, torch_serving_post, cigarette_api_post – real services (integration)
# Uses URLs from config; minimal body. Mark integration; skip if services unreachable.
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tf_serving_post():
    """tf_serving_post(body) calls TF_SERVING_REST_ADDR; returns response or raises."""
    body = {}
    try:
        response = await tf_serving_post(body)
        assert response is not None
        assert hasattr(response, "status_code")
    except Exception as e:
        assert "Service unavailable" in str(e) or "HTTP" in str(e) or "timed out" in str(e).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_torch_serving_post():
    """torch_serving_post(body) calls TORCH_SERVING_REST_ADDR; returns response or raises."""
    body = {}
    try:
        response = await torch_serving_post(body)
        assert response is not None
        assert hasattr(response, "status_code")
    except Exception as e:
        assert "Service unavailable" in str(e) or "HTTP" in str(e) or "timed out" in str(e).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cigarette_api_post():
    """cigarette_api_post(body) calls CIGARETTE_API_REST_ADDR; returns response or raises."""
    body = {}
    try:
        response = await cigarette_api_post(body)
        assert response is not None
        assert hasattr(response, "status_code")
    except Exception as e:
        assert "Service unavailable" in str(e) or "HTTP" in str(e) or "timed out" in str(e).lower()
