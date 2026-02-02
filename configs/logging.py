import time
import functools
import logging
import psutil
import os
import traceback

# Basic logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def simple_logger(log_level=logging.INFO):
    def decorator(func):
        logger = logging.getLogger(func.__name__)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())

            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            logger.log(log_level, f"CALLING: {func.__name__}")

            try:
                result = await func(*args, **kwargs)

                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024

                logger.log(
                    log_level,
                    f"FINISHED: {func.__name__} | "
                    f"Time: {end_time - start_time:.4f}s | "
                    f"Memory Δ: {end_memory - start_memory:+.2f} MB"
                )

                return result

            except Exception as e:
                logger.error(f"ERROR in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())

            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            logger.log(log_level, f"CALLING: {func.__name__}")

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024

                logger.log(
                    log_level,
                    f"FINISHED: {func.__name__} | "
                    f"Time: {end_time - start_time:.4f}s | "
                    f"Memory Δ: {end_memory - start_memory:+.2f} MB"
                )

                return result

            except Exception as e:
                logger.error(f"ERROR in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
