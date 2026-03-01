import time
import functools
import logging
import psutil
import os
import traceback
import asyncio
from contextvars import ContextVar
from configs.config import LOG_LEVEL


# Context Variable
request_id_var = ContextVar("request_id", default=None)


class _RequestIdFilter(logging.Filter):
    """Injects request_id from context var into every log record."""
    def filter(self, record):
        request_id = request_id_var.get()
        record.request_id_prefix = f"[request_id={request_id}] " if request_id else ""
        return True

# Creates a StreamHandler — this sends log output to stderr (the console)
_handler = logging.StreamHandler()
# Formats the log output to include the request_id prefix
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(request_id_prefix)s%(message)s"))
# Adds the request_id prefix to the log output
_handler.addFilter(_RequestIdFilter())
# Adds the handler to the root logger
logging.root.addHandler(_handler)
# Sets the log level for the root logger
logging.root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def simple_logger():
    def decorator(func):
        logger = logging.getLogger(func.__name__)
        log_level = getattr(logging, LOG_LEVEL, logging.INFO)

        def _start():
            start_time = time.perf_counter()
            start_memory = None

            if logger.isEnabledFor(logging.DEBUG):
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024

            logger.log(log_level, f"CALLING: {func.__name__}")
            return start_time, start_memory

        def _end(start_time, start_memory):
            elapsed = time.perf_counter() - start_time
            msg = f"FINISHED: {func.__name__} | Time: {elapsed:.4f}s"

            if logger.isEnabledFor(logging.DEBUG) and start_memory is not None:
                process = psutil.Process(os.getpid())
                end_memory = process.memory_info().rss / 1024 / 1024
                msg += f" | Memory Δ: {end_memory - start_memory:+.2f} MB"

            logger.log(log_level, msg)

        def _handle_exception(e):
            logger.info(f"ERROR in {func.__name__}: {e}")
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
