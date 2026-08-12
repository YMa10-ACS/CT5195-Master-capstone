"""Centralized metrics recording helpers for distributed VLM runs."""

import os
import time


class MetricsRecorder:
    """Collect per-request metrics without owning business control flow."""

    def __init__(self, initial=None, *, pid=None):
        self.values = dict(initial or {})
        self.pid = os.getpid() if pid is None else pid
        self._counters = {}
        self._cpu_counters = {}
        self._cpu_samplers = {}

    def set(self, name, value):
        self.values[name] = value

    def add(self, name, value):
        self.values[name] = self.values.get(name, 0) + value

    def update(self, values):
        if values:
            self.values.update(values)

    def get(self, name, default=None):
        return self.values.get(name, default)

    def __getitem__(self, name):
        return self.values[name]

    def __setitem__(self, name, value):
        self.values[name] = value

    def __contains__(self, name):
        return name in self.values

    def start_counter(self, name):
        self._counters[name] = time.perf_counter()

    def stop_counter(self, name):
        start = self._counters.pop(name, None)
        if start is None:
            raise KeyError(f"Metric counter {name!r} was not started")
        self.values[name] = time.perf_counter() - start
        return self.values[name]

    def add_counter(self, name):
        start = self._counters.pop(name, None)
        if start is None:
            raise KeyError(f"Metric counter {name!r} was not started")
        elapsed = time.perf_counter() - start
        self.values[name] = self.values.get(name, 0.0) + elapsed
        return elapsed

    def start_cpu_counter(self, name):
        self._cpu_counters[name] = self._process_cpu_seconds()

    def stop_cpu_counter(self, name):
        start = self._cpu_counters.pop(name, None)
        if start is None:
            raise KeyError(f"CPU counter {name!r} was not started")
        self.values[name] = self._process_cpu_seconds() - start
        return self.values[name]

    def start_cpu_sampling(self, label, interval=0.5):
        from monitor import ProcessSampler

        sampler = ProcessSampler(pid=self.pid, interval=interval)
        sampler.start()
        self._cpu_samplers[label] = sampler

    def stop_cpu_sampling(self, label, avg_metric, max_metric):
        sampler = self._cpu_samplers.pop(label, None)
        if sampler is None:
            raise KeyError(f"CPU sampler {label!r} was not started")
        sampler.stop()
        cpu_samples = [cpu for cpu, _ in sampler.samples]
        self.values[avg_metric] = (
            sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        )
        self.values[max_metric] = max(cpu_samples) if cpu_samples else 0.0
        return self.values[avg_metric], self.values[max_metric]

    def row(self):
        return dict(self.values)

    def _process_cpu_seconds(self):
        import psutil

        times = psutil.Process(self.pid).cpu_times()
        return times.user + times.system
