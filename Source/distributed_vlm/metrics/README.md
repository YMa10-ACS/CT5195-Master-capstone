# Metrics package

This package centralizes the distributed VLM experiment metrics:

- `names.py` defines metric names and the stable CSV column order.
- `timing.py` provides request-local timing and counter helpers.
- `csv_writer.py` writes per-request metrics and batch throughput rows.

Metric units follow these conventions:

- `*_time` and `*_latency`: seconds
- `*_pct`: percentage
- `*_bytes`: bytes
- `*_tokens`, `*_count`: counts

The CSV field names remain backward-compatible with existing experiment records.
The nanoVLM decoder receives a plain metrics dictionary and remains independent
from this application package.
