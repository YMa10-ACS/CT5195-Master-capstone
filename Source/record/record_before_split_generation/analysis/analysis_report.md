# Distributed VLM Record Analysis

## Dataset

- Metrics rows: 60
- Transport modes: http, https-aes-gcm, https-chacha20
- Targets: 10.128.0.6 (Confidential VM / TDX), 10.128.0.7 (Normal VM)
- Metrics rows per file: 10

## Key Findings

- Fastest mean split latency: https-aes-gcm on Normal VM (10.128.0.7), 33.338 s.
- Slowest mean split latency: http on Confidential VM / TDX (10.128.0.6), 34.947 s.
- Best throughput: https-aes-gcm on Normal VM (1.800 images/min, 33.341 s/image).
- Lowest throughput: http on Confidential VM / TDX (1.717 images/min, 34.949 s/image).

## Mean Latency By Transport And Target

| transport | cloud type | split s | edge embed s | transfer s | cloud request s | decoder generate s | prefill s | decode step s | network overhead ms | cloud CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| http | Confidential VM / TDX | 34.947 | 18.297 | 16.543 | 16.539 | 16.529 | 1.334 | 13.620 | 3.477 | 33.092 |
| http | Normal VM | 33.357 | 18.299 | 14.949 | 14.946 | 14.938 | 1.272 | 12.195 | 3.352 | 29.900 |
| https-aes-gcm | Confidential VM / TDX | 34.754 | 18.309 | 16.337 | 16.332 | 16.322 | 1.298 | 13.446 | 4.506 | 32.676 |
| https-aes-gcm | Normal VM | 33.338 | 18.287 | 14.944 | 14.940 | 14.931 | 1.314 | 12.148 | 4.267 | 29.888 |
| https-chacha20 | Confidential VM / TDX | 34.777 | 18.291 | 16.378 | 16.374 | 16.364 | 1.325 | 13.458 | 4.524 | 32.758 |
| https-chacha20 | Normal VM | 33.406 | 18.319 | 14.977 | 14.973 | 14.965 | 1.298 | 12.197 | 4.227 | 29.955 |

## TDX Overhead Compared With Normal VM

| transport | paired images | split delta s | split delta % | cloud request delta s | cloud request delta % | cloud CPU delta s | cloud CPU delta % |
|---|---:|---:|---:|---:|---:|---:|---:|
| http | 10 | 1.590 | 4.890% | 1.593 | 10.745% | 3.192 | 10.759% |
| https-aes-gcm | 10 | 1.416 | 4.345% | 1.392 | 9.379% | 2.788 | 9.387% |
| https-chacha20 | 10 | 1.372 | 4.167% | 1.400 | 9.372% | 2.803 | 9.376% |

## HTTPS Overhead Compared With HTTP

| cloud type | transport | split delta s | transfer delta s | network overhead delta ms | cloud request delta s |
|---|---:|---:|---:|---:|---:|
| Confidential VM / TDX | https-aes-gcm | -0.193 | -0.206 | 1.029 | -0.207 |
| Confidential VM / TDX | https-chacha20 | -0.170 | -0.165 | 1.047 | -0.166 |
| Normal VM | https-aes-gcm | -0.018 | -0.005 | 0.914 | -0.006 |
| Normal VM | https-chacha20 | 0.049 | 0.028 | 0.875 | 0.027 |

## Generated Files

- metrics_combined.csv: one row per image request, with cloud_type and image_id added.
- metrics_summary_by_transport_target.csv: mean/median/std/min/max by transport and target.
- throughput_combined.csv: one row per throughput run, with seconds_per_image added.
- tdx_overhead_vs_normal_vm.csv: paired-image TDX minus Normal VM deltas.
- transport_overhead_vs_http.csv: HTTPS mode deltas against HTTP within the same target.
