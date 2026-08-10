"""Small timing helpers that operate on a request-local metrics dictionary."""

import time
from contextlib import contextmanager


def add_timing(metrics, metric_name, elapsed):
    if metrics is not None:
        metrics[metric_name] = metrics.get(metric_name, 0.0) + elapsed


def add_count(metrics, metric_name, count):
    if metrics is not None:
        metrics[metric_name] = metrics.get(metric_name, 0) + count


@contextmanager
def measure(metrics, metric_name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        add_timing(metrics, metric_name, elapsed)
        print(f"[timing] {metric_name}: {elapsed} seconds")
