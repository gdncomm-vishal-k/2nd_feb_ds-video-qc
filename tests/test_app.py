"""Tests for FastAPI application endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Import app after path is set (conftest)
from application.app import (
    app,
    health_check,
    check_service,
    dependent_services_health_check,
    predict,
)


def test_health_check():
    client = TestClient(app)
    r = client.get("/sys-info/health")
    assert r.status_code == 200
    assert r.json() == {"Status": "Healthy"}


class _AsyncCtx:
    def __init__(self, status):
        self._resp = type("R", (), {"status": status})()

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass


def _make_session(get_ctx):
    return type("S", (), {"get": lambda self, url, timeout=5: get_ctx})()


@pytest.mark.asyncio
async def test_check_service_returns_true_on_200():
    with patch("application.app.aiohttp.ClientSession") as mock_session:
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=_make_session(_AsyncCtx(200)))
        session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = session_ctx
        result = await check_service("http://example.com/health")
        assert result is True


@pytest.mark.asyncio
async def test_check_service_returns_false_on_non_200():
    with patch("application.app.aiohttp.ClientSession") as mock_session:
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=_make_session(_AsyncCtx(500)))
        session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = session_ctx
        result = await check_service("http://example.com/health")
        assert result is False


@pytest.mark.asyncio
async def test_check_service_raises_on_exception():
    with patch("application.app.aiohttp.ClientSession") as mock_session:
        class FailCtx:
            async def __aenter__(self):
                raise Exception("network error")
            async def __aexit__(self, *args):
                pass
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=_make_session(FailCtx()))
        session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = session_ctx
        with pytest.raises(Exception) as exc_info:
            await check_service("http://example.com/health")
        assert "network error" in str(exc_info.value) or "Error checking" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dependent_services_health_check():
    with patch("application.app.check_service", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        result = await dependent_services_health_check()
        assert result["cigarette_api"] is True
        assert result["tf_serving"] is True
        assert result["torch_serving"] is True


@pytest.mark.asyncio
async def test_dependent_services_health_check_mixed():
    with patch("application.app.check_service", new_callable=AsyncMock) as mock_check:
        mock_check.side_effect = [True, False, True]
        result = await dependent_services_health_check()
        assert result["cigarette_api"] is True
        assert result["tf_serving"] is False
        assert result["torch_serving"] is True


@pytest.mark.asyncio
async def test_predict_success():
    from schemas.schemas import VideoQCRequest

    request = VideoQCRequest(
        sku_id=["s1"],
        caption="Hi",
        video_id="v1",
        video_path="https://example.com/v.mp4",
    )
    mock_response = {
        "sku_id": ["s1"],
        "caption": "Hi",
        "video_id": "v1",
        "video_path": "https://example.com/v.mp4",
        "video_qc": {},
        "audio_qc": {"rejected": False, "rejected_reason": ""},
        "caption_qc": {"rejected": False, "rejected_reason": ""},
    }
    with patch("application.app.dependent_services_health_check", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "cigarette_api": True,
            "tf_serving": True,
            "torch_serving": True,
        }
        with patch("application.app.run_video_qc_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = mock_response
            result = await predict(request)
            assert result["sku_id"] == ["s1"]
            assert result["video_id"] == "v1"


@pytest.mark.asyncio
async def test_predict_unhealthy_services_raises_500():
    from fastapi import HTTPException
    from schemas.schemas import VideoQCRequest

    request = VideoQCRequest(
        sku_id=["s1"],
        caption="Hi",
        video_id="v1",
        video_path="https://example.com/v.mp4",
    )
    with patch("application.app.dependent_services_health_check", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "cigarette_api": True,
            "tf_serving": False,
            "torch_serving": True,
        }
        with pytest.raises(HTTPException) as exc_info:
            await predict(request)
        assert exc_info.value.status_code == 500
        assert "unhealthy" in exc_info.value.detail.lower() or "dependent" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_predict_pipeline_exception_raises_500():
    from fastapi import HTTPException
    from schemas.schemas import VideoQCRequest

    request = VideoQCRequest(
        sku_id=["s1"],
        caption="Hi",
        video_id="v1",
        video_path="https://example.com/v.mp4",
    )
    with patch("application.app.dependent_services_health_check", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "cigarette_api": True,
            "tf_serving": True,
            "torch_serving": True,
        }
        with patch("application.app.run_video_qc_pipeline", new_callable=AsyncMock, side_effect=RuntimeError("pipeline failed")):
            with pytest.raises(HTTPException) as exc_info:
                await predict(request)
            assert exc_info.value.status_code == 500


def test_predict_endpoint_integration():
    client = TestClient(app)
    with patch("application.app.dependent_services_health_check", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "cigarette_api": True,
            "tf_serving": True,
            "torch_serving": True,
        }
        with patch("application.app.run_video_qc_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = {
                "sku_id": ["s1"],
                "caption": "Hi",
                "video_id": "v1",
                "video_path": "https://example.com/v.mp4",
                "video_qc": {},
                "audio_qc": {"rejected": False, "rejected_reason": ""},
                "caption_qc": {"rejected": False, "rejected_reason": ""},
            }
            r = client.post(
                "/predict",
                json={
                    "sku_id": ["s1"],
                    "caption": "Hi",
                    "video_id": "v1",
                    "video_path": "https://example.com/v.mp4",
                },
            )
            assert r.status_code == 200
            assert r.json()["video_id"] == "v1"
