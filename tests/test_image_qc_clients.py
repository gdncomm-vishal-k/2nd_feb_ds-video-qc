"""Tests for image_qc_utils.clients."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from components.image_qc_utils.clients import (
    post,
    tf_serving_post,
    torch_serving_post,
    cigarette_api_post,
)


@pytest.mark.asyncio
async def test_post_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("components.image_qc_utils.clients.httpx.AsyncClient", return_value=mock_client):
        result = await post("http://example.com", {"key": "value"})
        assert result == mock_resp


@pytest.mark.asyncio
async def test_post_connect_error():
    with patch("components.image_qc_utils.clients.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("conn err"))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(Exception) as exc_info:
            await post("http://example.com", {})
        assert "unavailable" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_post_timeout():
    with patch("components.image_qc_utils.clients.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(Exception):
            await post("http://example.com", {})


@pytest.mark.asyncio
async def test_post_http_status_error():
    with patch("components.image_qc_utils.clients.httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=mock_resp))
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(Exception) as exc_info:
            await post("http://example.com", {})
        assert "500" in str(exc_info.value) or "HTTP" in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_generic_exception():
    with patch("components.image_qc_utils.clients.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("generic err"))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(Exception) as exc_info:
            await post("http://example.com", {})
        assert "generic err" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tf_serving_post_calls_post():
    with patch("components.image_qc_utils.clients.post", new_callable=AsyncMock) as mock_post:
        await tf_serving_post({"images": []})
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_torch_serving_post_calls_post():
    with patch("components.image_qc_utils.clients.post", new_callable=AsyncMock) as mock_post:
        await torch_serving_post({"images": []})
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_cigarette_api_post_calls_post():
    with patch("components.image_qc_utils.clients.post", new_callable=AsyncMock) as mock_post:
        await cigarette_api_post({})
        mock_post.assert_called_once()
