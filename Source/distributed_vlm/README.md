<!--
 * @Description: 
 * @Date: 2026-08-18 14:14:45
 * @Author: Yaoquan Ma
-->
# Distributed VLM Application

This directory contains the project-specific implementation of secure
edge/cloud split inference. It includes the edge and cloud entry points,
transport and attestation components, metrics utilities, execution scripts and
historical experiment records.

## Main programs

| File or directory | Responsibility |
| --- | --- |
| `edge.py` | Image preprocessing, visual encoding, local baseline and split-inference client |
| `cloud.py` | Flask application that reconstructs embeddings and generates text |
| `vlm.py` | Shared nanoVLM loading, prompt construction and generation helpers |
| `transfer.py` | HTTP/TLS embedding transfer client and cloud response validation |
| `attestation.py` | Attestation challenge, evidence creation and verification |
| `monitor.py` | Process CPU and memory sampling |
| `metrics/` | Metric names, recorders and CSV writers |
| `tdx_attest/` | Go helper for creating and verifying Intel TDX evidence |
| `test_scripts/` | Scripts for functional validation and performance benchmarking |

`vlm.py`, `transfer.py`, `attestation.py`, `monitor.py`, `metrics/` and
`tdx_attest/` are infrastructure components rather than experiment business
logic. They can later be moved into a dedicated `framework/` package after
imports and tests have been prepared.

## Execution scripts
In test_scripts directory:
- `test_scripts/start_cloud.sh` starts one cloud service for a selected transport.
- `test_scripts/run_edge_comparison.sh` runs an edge batch and writes metrics.
- `test_scripts/run_all_modes.sh` compares normal VM, TDX VM and local inference modes.
- `test_scripts/run_all_cloud_modes.sh` starts the supported cloud transport modes.
- `test_scripts/check_tls_version.sh` checks the negotiated TLS configuration.
- `init.sh` prepares the application environment and nanoVLM dependency.

## Historical records

- `record_before_split_generation/` contains results from the earlier
  generation implementation.
- `record_split_generation/` contains results after splitting generation.
- `record_split_generate_function/` contains results after extracting the
  split generation function.

Each record directory includes raw logs/CSV files and derived files under its
`analysis/` directory. These names describe implementation stages; future runs
should use timestamped immutable run directories with a configuration manifest
and Git commit identifier.

## Configuration inputs

The application currently receives configuration from command-line arguments,
environment variables and shell-script defaults. Important inputs include:

- model weights and prompt;
- input image or dataset directory;
- cloud target and port;
- local or split inference mode;
- HTTP or TLS transport;
- attestation state and TDX helper path;
- certificate path;
- metrics and throughput output paths.

## Metrics

See `metrics/README.md` for metric names, units and the preferred recording
API. Existing CSV column names are kept stable for comparison with historical
experiments.
