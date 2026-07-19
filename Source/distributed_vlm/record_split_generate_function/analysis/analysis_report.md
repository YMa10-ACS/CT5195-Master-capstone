# Distributed VLM Record Analysis

## Dataset

- Metrics rows: 60
- Transport modes: http, https-aes-gcm, https-chacha20
- Targets: 10.128.0.6 (Confidential VM / TDX), 10.128.0.7 (Normal VM)
- Metrics rows per file: 10

## Key Findings

- Fastest mean split latency: https-chacha20 on Normal VM (10.128.0.7), 23.367 s.
- Slowest mean split latency: http on Confidential VM / TDX (10.128.0.6), 23.756 s.
- Best throughput: https-chacha20 on Normal VM (2.567 images/min, 23.370 s/image).
- Lowest throughput: http on Confidential VM / TDX (2.525 images/min, 23.759 s/image).

## Mean Latency By Transport And Target

| transport | cloud type | split s | edge embed s | transfer s | cloud request s | generated tokens | decoder generate s | prefill s | decode step s | decode ms/token | KV cat s | KV repeat s | attn core s | MLP s | network overhead ms | cloud CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| http | Confidential VM / TDX | 23.756 | 18.310 | 5.337 | 5.334 | 82.4 | 5.324 | 1.332 | 3.552 | 43.112 | 0.366 | 0.456 | 0.595 | 2.007 | 3.432 | 10.662 |
| http | Normal VM | 23.375 | 18.389 | 4.878 | 4.875 | 82.4 | 4.867 | 1.271 | 3.187 | 38.683 | 0.353 | 0.424 | 0.567 | 1.842 | 3.169 | 9.748 |
| https-aes-gcm | Confidential VM / TDX | 23.750 | 18.315 | 5.326 | 5.322 | 82.4 | 5.311 | 1.326 | 3.548 | 43.057 | 0.365 | 0.453 | 0.598 | 1.997 | 4.386 | 10.641 |
| https-aes-gcm | Normal VM | 23.374 | 18.332 | 4.933 | 4.929 | 82.4 | 4.920 | 1.312 | 3.199 | 38.819 | 0.355 | 0.423 | 0.571 | 1.870 | 4.153 | 9.856 |
| https-chacha20 | Confidential VM / TDX | 23.750 | 18.295 | 5.347 | 5.342 | 82.4 | 5.332 | 1.327 | 3.566 | 43.278 | 0.366 | 0.453 | 0.599 | 2.006 | 4.365 | 10.681 |
| https-chacha20 | Normal VM | 23.367 | 18.334 | 4.924 | 4.920 | 82.4 | 4.912 | 1.290 | 3.212 | 38.980 | 0.353 | 0.428 | 0.570 | 1.858 | 4.116 | 9.838 |

## TDX Overhead Compared With Normal VM

| transport | paired images | split delta s | split delta % | cloud request delta s | cloud request delta % | cloud CPU delta s | cloud CPU delta % |
|---|---:|---:|---:|---:|---:|---:|---:|
| http | 10 | 0.381 | 1.607% | 0.459 | 9.198% | 0.914 | 9.136% |
| https-aes-gcm | 10 | 0.377 | 1.612% | 0.393 | 7.738% | 0.785 | 7.729% |
| https-chacha20 | 10 | 0.383 | 1.635% | 0.423 | 8.273% | 0.843 | 8.249% |

## HTTPS Overhead Compared With HTTP

| cloud type | transport | split delta s | transfer delta s | network overhead delta ms | cloud request delta s |
|---|---:|---:|---:|---:|---:|
| Confidential VM / TDX | https-aes-gcm | -0.006 | -0.011 | 0.954 | -0.012 |
| Confidential VM / TDX | https-chacha20 | -0.006 | 0.009 | 0.933 | 0.008 |
| Normal VM | https-aes-gcm | -0.002 | 0.055 | 0.984 | 0.054 |
| Normal VM | https-chacha20 | -0.008 | 0.046 | 0.947 | 0.045 |

## Generated Files

- metrics_combined.csv: one row per image request, with cloud_type and image_id added.
- metrics_summary_by_transport_target.csv: mean/median/std/min/max by transport and target.
- throughput_combined.csv: one row per throughput run, with seconds_per_image added.
- tdx_overhead_vs_normal_vm.csv: paired-image TDX minus Normal VM deltas.
- transport_overhead_vs_http.csv: HTTPS mode deltas against HTTP within the same target.
