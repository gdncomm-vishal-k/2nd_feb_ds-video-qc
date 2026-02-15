import time
import functools
import logging
import psutil
import os
import traceback
import asyncio
from configs.config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def simple_logger(log_level=logging.INFO, request_id=None):
    def decorator(func):
        logger = logging.getLogger(func.__name__)

        def _prefix():
            return f"[request_id={request_id}] " if request_id else ""

        def _start():
            start_time = time.time()
            start_memory = None

            if logger.isEnabledFor(logging.DEBUG):
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024

            logger.log(
                log_level,
                f"{_prefix()}CALLING: {func.__name__}"
            )
            return start_time, start_memory

        def _end(start_time, start_memory):
            elapsed = time.time() - start_time
            msg = f"{_prefix()}FINISHED: {func.__name__} | Time: {elapsed:.4f}s"

            if logger.isEnabledFor(logging.DEBUG) and start_memory is not None:
                process = psutil.Process(os.getpid())
                end_memory = process.memory_info().rss / 1024 / 1024
                msg += f" | Memory Δ: {end_memory - start_memory:+.2f} MB"

            logger.log(log_level, msg)

        def _handle_exception(e):
            logger.info(
                f"{_prefix()}ERROR in {func.__name__}: {e}"
            )
            logger.info(traceback.format_exc())

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time, start_memory = _start()
            try:
                result = await func(*args, **kwargs)
                _end(start_time, start_memory)
                return result
            except Exception as e:
                _handle_exception(e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time, start_memory = _start()
            try:
                result = func(*args, **kwargs)
                _end(start_time, start_memory)
                return result
            except Exception as e:
                _handle_exception(e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator