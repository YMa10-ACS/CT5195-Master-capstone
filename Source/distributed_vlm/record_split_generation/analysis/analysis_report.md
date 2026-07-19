# Distributed VLM Record Analysis

## Dataset

- Metrics rows: 60
- Transport modes: http, https-aes-gcm, https-chacha20
- Targets: 10.128.0.6 (Confidential VM / TDX), 10.128.0.7 (Normal VM)
- Metrics rows per file: 10

## Key Findings

- Fastest mean split latency: http on Normal VM (10.128.0.7), 23.304 s.
- Slowest mean split latency: https-aes-gcm on Confidential VM / TDX (10.128.0.6), 23.800 s.
- Best throughput: http on Normal VM (2.574 images/min, 23.307 s/image).
- Lowest throughput: https-aes-gcm on Confidential VM / TDX (2.521 images/min, 23.803 s/image).

## Mean Latency By Transport And Target

| transport | cloud type | split s | edge embed s | transfer s | cloud request s | generated tokens | decoder generate s | prefill s | decode step s | decode ms/token | network overhead ms | cloud CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| http | Confidential VM / TDX | 23.742 | 18.289 | 5.345 | 5.341 | 82.4 | 5.333 | 1.359 | 3.531 | 42.857 | 3.405 | 10.679 |
| http | Normal VM | 23.304 | 18.321 | 4.874 | 4.871 | 82.4 | 4.863 | 1.281 | 3.174 | 38.521 | 3.113 | 9.738 |
| https-aes-gcm | Confidential VM / TDX | 23.800 | 18.335 | 5.357 | 5.353 | 82.4 | 5.344 | 1.322 | 3.582 | 43.470 | 4.331 | 10.704 |
| https-aes-gcm | Normal VM | 23.373 | 18.349 | 4.914 | 4.910 | 82.4 | 4.901 | 1.308 | 3.184 | 38.640 | 4.173 | 9.816 |
| https-chacha20 | Confidential VM / TDX | 23.743 | 18.340 | 5.293 | 5.289 | 82.4 | 5.280 | 1.317 | 3.524 | 42.769 | 4.274 | 10.575 |
| https-chacha20 | Normal VM | 23.312 | 18.334 | 4.868 | 4.864 | 82.4 | 4.855 | 1.281 | 3.167 | 38.431 | 4.094 | 9.724 |

## TDX Overhead Compared With Normal VM

| transport | paired images | split delta s | split delta % | cloud request delta s | cloud request delta % | cloud CPU delta s | cloud CPU delta % |
|---|---:|---:|---:|---:|---:|---:|---:|
| http | 10 | 0.437 | 1.867% | 0.471 | 9.383% | 0.941 | 9.378% |
| https-aes-gcm | 10 | 0.427 | 1.864% | 0.443 | 8.737% | 0.888 | 8.769% |
| https-chacha20 | 10 | 0.431 | 1.859% | 0.425 | 8.542% | 0.851 | 8.547% |

## HTTPS Overhead Compared With HTTP

| cloud type | transport | split delta s | transfer delta s | network overhead delta ms | cloud request delta s |
|---|---:|---:|---:|---:|---:|
| Confidential VM / TDX | https-aes-gcm | 0.058 | 0.012 | 0.926 | 0.011 |
| Confidential VM / TDX | https-chacha20 | 0.001 | -0.052 | 0.869 | -0.053 |
| Normal VM | https-aes-gcm | 0.068 | 0.040 | 1.060 | 0.039 |
| Normal VM | https-chacha20 | 0.007 | -0.006 | 0.981 | -0.007 |

## Generated Files

- metrics_combined.csv: one row per image request, with cloud_type and image_id added.
- metrics_summary_by_transport_target.csv: mean/median/std/min/max by transport and target.
- throughput_combined.csv: one row per throughput run, with seconds_per_image added.
- tdx_overhead_vs_normal_vm.csv: paired-image TDX minus Normal VM deltas.
- transport_overhead_vs_http.csv: HTTPS mode deltas against HTTP within the same target.
