"""
Tests for src/components/pub_sub.py.
Minimal mocks: only Kafka consumer/producer (so no broker) and produce_and_commit (so no send).
process_messages logic is tested with real JSON and real async process_batch_requests_from_kafka.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from configs.kafka_config import consumer_topics, producer_topic
from src.components.pub_sub import PubSub


def _make_msg(value: bytes, offset: int = 0):
    """Fake Kafka message with .value and .offset."""
    m = MagicMock()
    m.value = value
    m.offset = offset
    return m


# ---------------------------------------------------------------------------
# process_messages – with minimal mocks (no real Kafka I/O)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_process_messages_empty_returns_early():
    """process_messages({}) returns without calling produce_and_commit."""
    mock_batch = AsyncMock(return_value=[])
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        pubsub = PubSub(consumer_topics, producer_topic, mock_batch)
        pubsub.produce_and_commit = AsyncMock()

        await pubsub.process_messages({})

        pubsub.produce_and_commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_messages_empty_partition_skipped():
    """process_messages({tp: []}) does not call produce_and_commit."""
    mock_batch = AsyncMock(return_value=[])
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        pubsub = PubSub(consumer_topics, producer_topic, mock_batch)
        pubsub.produce_and_commit = AsyncMock()

        tp = MagicMock()
        await pubsub.process_messages({tp: []})

        pubsub.produce_and_commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_messages_valid_json_calls_batch_and_produce():
    """process_messages with valid JSON decodes, calls batch processor, then produce_and_commit."""
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        async def fake_batch(requests):
            return [{"processed": r.get("video_id", "unknown")} for r in requests]

        pubsub = PubSub(consumer_topics, producer_topic, fake_batch)
        pubsub.produce_and_commit = AsyncMock()

        body = {"video_id": "test-123", "video_path": "https://example.com/v.mp4", "sku_id": ["A"], "caption": ""}
        tp = MagicMock()
        messages = {tp: [_make_msg(json.dumps(body).encode("utf-8"), 0)]}

        await pubsub.process_messages(messages)

        pubsub.produce_and_commit.assert_called_once()
        call_args = pubsub.produce_and_commit.call_args
        assert call_args[0][2] == [{"processed": "test-123"}]


@pytest.mark.asyncio
async def test_process_messages_invalid_json_returns_decode_error_response():
    """process_messages with invalid JSON puts error in responses and still calls produce_and_commit."""
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        async def fake_batch(requests):
            return [{"ok": True}]  # only valid requests

        pubsub = PubSub(consumer_topics, producer_topic, fake_batch)
        pubsub.produce_and_commit = AsyncMock()

        tp = MagicMock()
        messages = {
            tp: [
                _make_msg(b"not valid json", 0),
            ]
        }

        await pubsub.process_messages(messages)

        pubsub.produce_and_commit.assert_called_once()
        responses = pubsub.produce_and_commit.call_args[0][2]
        assert responses == [{"error": "Invalid JSON format"}]


@pytest.mark.asyncio
async def test_process_messages_batch_exception_returns_error_per_request():
    """When process_batch_requests_from_kafka raises, responses get error payloads."""
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        async def failing_batch(requests):
            raise RuntimeError("service down")

        pubsub = PubSub(consumer_topics, producer_topic, failing_batch)
        pubsub.produce_and_commit = AsyncMock()

        body = {"video_id": "v1", "video_path": "https://x.co/v.mp4", "sku_id": ["S"], "caption": ""}
        tp = MagicMock()
        messages = {tp: [_make_msg(json.dumps(body).encode("utf-8"), 0)]}

        await pubsub.process_messages(messages)

        pubsub.produce_and_commit.assert_called_once()
        responses = pubsub.produce_and_commit.call_args[0][2]
        assert len(responses) == 1
        assert responses[0].get("error") == "Batch processing failed"
        assert responses[0].get("input_received") == body


@pytest.mark.asyncio
async def test_process_messages_mixed_valid_invalid_json():
    """One valid and one invalid JSON message: valid gets batch result, invalid gets decode error."""
    with patch("src.components.pub_sub.AIOKafkaConsumer", MagicMock()), patch(
        "src.components.pub_sub.AIOKafkaProducer", MagicMock()
    ):
        async def fake_batch(requests):
            assert len(requests) == 1
            return [{"done": requests[0]["video_id"]}]

        pubsub = PubSub(consumer_topics, producer_topic, fake_batch)
        pubsub.produce_and_commit = AsyncMock()

        body = {"video_id": "only-valid", "video_path": "https://x.co/v.mp4", "sku_id": ["S"], "caption": ""}
        tp = MagicMock()
        messages = {
            tp: [
                _make_msg(b"{invalid", 0),
                _make_msg(json.dumps(body).encode("utf-8"), 1),
            ]
        }

        await pubsub.process_messages(messages)

        pubsub.produce_and_commit.assert_called_once()
        responses = pubsub.produce_and_commit.call_args[0][2]
        assert responses[0] == {"error": "Invalid JSON format"}
        assert responses[1] == {"done": "only-valid"}
