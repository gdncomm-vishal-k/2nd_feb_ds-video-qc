"""
BEGINNER-FRIENDLY TESTS FOR app.py
==================================
Import the API functions once at the top, then call them in each test and assert.
"""

from unittest.mock import patch

import pytest

# Import what we need from the app and schemas (once at the top)
from application.app import (
    health_check,
    dependent_services_health_check,
    on_startup,
    on_app_exit,
    predict,
)
from conftest import MockPubSub
from schemas.schemas import VideoQCRequest


# ---------------------------------------------------------------------------
# TEST 1: Health check
# ---------------------------------------------------------------------------
def test_health_check():
    result = health_check()
    assert result == {"Status": "Healthy"}


# ---------------------------------------------------------------------------
# TEST 2: Dependent services health (async – use await)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dependent_services_health_check():
    result = await dependent_services_health_check()
    assert "cigarette_api" in result and "tf_serving" in result and "torch_serving" in result
    print(result)
    assert isinstance(result["cigarette_api"], bool)
    assert isinstance(result["tf_serving"], bool)
    assert isinstance(result["torch_serving"], bool)
    print(result["cigarette_api"], result["tf_serving"], result["torch_serving"])


# ---------------------------------------------------------------------------
# TEST 3: Predict (async; slow – real pipeline, skip with: pytest -m "not integration")
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_predict_success(sample_video_qc_request):
    request = VideoQCRequest(**sample_video_qc_request)
    response = await predict(request)
    print(response)
    # predict() returns a dict (from pipeline), not a Pydantic model – use dict keys
    assert response["video_id"] == sample_video_qc_request["video_id"]
    assert response["sku_id"] == sample_video_qc_request["sku_id"]
    assert "video_qc" in response and "audio_qc" in response and "caption_qc" in response
    assert "rejected" in response["audio_qc"] and "rejected" in response["caption_qc"]


# ---------------------------------------------------------------------------
# TEST 4: on_startup and on_app_exit (lifespan events)
# ---------------------------------------------------------------------------
# These run when the app starts/stops (e.g. with TestClient). We test them directly:
# patch PubSub so no real Kafka; call on_startup() then on_app_exit(); no exception = pass.
@pytest.mark.asyncio
async def test_on_startup_and_on_app_exit():
    with patch("application.app.pub_sub.PubSub", MockPubSub):
        await on_startup()
        await on_app_exit()


# ---------------------------------------------------------------------------
# OPTIONAL: Same health check via HTTP (TestClient from conftest)
# ---------------------------------------------------------------------------
def test_health_check_via_http(client):
    response = client.get("/sys-info/health")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"Status": "Healthy"}
