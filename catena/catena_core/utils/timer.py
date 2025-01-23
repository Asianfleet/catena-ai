from typing import Optional
from functools import wraps
from time import perf_counter


class Timer:
    """Timer class for timing code execution"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.time_elapsed: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.time_elapsed is not None:
            return self.time_elapsed
        elif self.start_time is not None:
            return perf_counter() - self.start_time
        else:
            return 0.0

    def start(self) -> float:
        self.start_time = perf_counter()
        return self.start_time

    def stop(self) -> float:
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.time_elapsed = self.end_time - self.start_time
        return self.end_time

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.time_elapsed = self.end_time - self.start_time

def record_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        time_elapsed = end_time - start_time
        wrapper.time_elapsed = time_elapsed
        return result
    return wrapper