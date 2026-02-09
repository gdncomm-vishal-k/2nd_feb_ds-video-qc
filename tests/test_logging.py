"""Tests for configs.logging (simple_logger decorator)."""
import pytest
from unittest.mock import patch
import asyncio

from configs.logging import simple_logger


def test_simple_logger_sync():
    @simple_logger()
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3


def test_simple_logger_sync_raises():
    @simple_logger()
    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError) as exc_info:
        fail()
    assert "expected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_simple_logger_async():
    @simple_logger()
    async def add_async(a, b):
        return a + b

    result = await add_async(1, 2)
    assert result == 3


@pytest.mark.asyncio
async def test_simple_logger_async_raises():
    @simple_logger()
    async def fail_async():
        raise RuntimeError("async err")

    with pytest.raises(RuntimeError) as exc_info:
        await fail_async()
    assert "async err" in str(exc_info.value)
