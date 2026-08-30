# Experiment Analysis

This directory contains scripts for aggregating and visualizing distributed
VLM experiment results.

## Files

- `analyze_record.py` combines per-request metric CSV files and throughput
  files, compares TDX with normal VMs, and writes a Markdown report.
- `plot_latency.R` summarizes latency measurements and creates a comparison
  plot.
- `plot_cpu_time.R` summarizes process CPU measurements and creates a
  comparison plot.

## Generated outputs

Depending on the script, analysis can produce:

- `metrics_combined.csv`;
- `metrics_summary_by_transport_target.csv`;
- `throughput_combined.csv`;
- `tdx_overhead_vs_normal_vm.csv`;
- `transport_overhead_vs_http.csv`;
- `analysis_report.md`;
- summary CSV files and PNG plots.

Generated analysis belongs with the experiment run from which it was derived.
Do not overwrite the raw CSV and log files.

## Current path limitation

These scripts were moved from `distributed_vlm/` without changing their path
configuration:

- `analyze_record.py` currently expects a `record/` directory beside itself.
- `plot_cpu_time.R` currently expects `analysis/record/`.
- `plot_latency.R` currently expects `analysis/record_all/record_tdx/`.

Passable input/output directories should be added as explicit command-line
arguments before treating these scripts as location-independent. Until then,
review and adjust the paths locally before running an analysis.
