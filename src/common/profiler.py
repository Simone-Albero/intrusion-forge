import time
import logging
import functools

logger = logging.getLogger(__name__)

def time_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        logger.info(f"TIME PROFILER {func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper
