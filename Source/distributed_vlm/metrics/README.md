# Metrics package

This package centralizes the distributed VLM experiment metrics:

- `names.py` defines metric names and the stable CSV column order.
- `recorder.py` provides `MetricsRecorder`, the preferred centralized API for new metrics.
- `csv_writer.py` writes per-request metrics and batch throughput rows.

Metric units follow these conventions:

- `*_time` and `*_latency`: seconds
- `*_pct`: percentage
- `*_bytes`: bytes
- `*_tokens`, `*_count`: counts

The CSV field names remain backward-compatible with existing experiment records.
The nanoVLM decoder receives a plain metrics dictionary and remains independent
from this application package.

Preferred recorder style for new code:

```python
metrics = MetricsRecorder({metric_names.TRANSPORT: transport})
metrics.start_counter(metric_names.TRANSFER_LATENCY)
# work
metrics.stop_counter(metric_names.TRANSFER_LATENCY)
metrics.add(metric_names.CLOUD_GENERATE_TOKENS, token_count)
write_metrics_csv(path, metrics.row())
```

Use `start_cpu_counter` / `stop_cpu_counter` for process CPU seconds and
`start_cpu_sampling` / `stop_cpu_sampling` for sampled process CPU percentages.
