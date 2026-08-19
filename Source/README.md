# Secure Distributed VLM

This project implements secure split inference for a Vision-Language Model (VLM). Image processing is performed on the edge, while language generation runs locally or on a remote cloud VM.

The experiments compare normal and Intel TDX confidential VMs using HTTP and TLS transport modes.

## System Workflow

1. The edge node processes an input image and generates a visual embedding.
2. The edge node optionally verifies the TDX environment through remote attestation.
3. The visual embedding is transferred to the cloud.
4. The cloud completes language generation and returns the result.
5. The system records latency, throughput and resource usage metrics.

## Directory Structure

- distributed_vlm/: edge and cloud applications and supporting components.
- nanoVLM/: the project fork of the nanoVLM framework.
- test_scripts/: scripts for starting services and running experiments.
- tdx_attest/: the Go helper for generating and verifying TDX evidence.
- configs/: OpenSSL configurations for the TLS experiments.
- secrets/: experimental TLS certificate and private key.
- datasets/: dataset download script and experiment images.
- record/: experiment logs, metrics and throughput results.
- analysis/: scripts for processing and plotting experiment results.
- env/: normal VM and TDX VM environment inspection results.
- bechmarks/: normal VM and TDX VM benchmark results.

## 1. Initialize the Environment

Run the initialization script from the repository root:

```bash
source Source/init.sh
```

The script:

- creates and activates `Source/.venv`;
- installs the required Python packages;
- installs Gunicorn;
- clones `YMa10-ACS/nanoVLM` into `Source/nanoVLM` when it is missing.

### About project fork from nanoVLM repo

The project fork of nanoVLM is required because it includes the decoder timing metrics used by `distributed_vlm/vlm.py`.

## 2. Download Experiment Images

The following command downloads images from the default AI2D dataset subset:

```bash
python Source/datasets/download_hf_images.py \
  --prefix Source/datasets \
  --name ai2d \
  --limit 20
```

The images are saved under `Source/datasets/ai2d/`.

## 3. Start the Cloud Services

Run the cloud scripts on the corresponding cloud VM.

### Normal VM

Remote attestation is disabled on the normal VM:

```bash
ATTESTATION=disabled bash Source/test_scripts/run_all_cloud_modes.sh
```

### TDX VM

Remote attestation is enabled on the TDX VM:

```bash
ATTESTATION=enabled bash Source/test_scripts/run_all_cloud_modes.sh
```

This starts the three transport modes:

- HTTP on port 8000;
- HTTPS with AES-GCM on port 8443;
- HTTPS with ChaCha20-Poly1305 on port 8444.

To start only one mode, use:

```bash
ATTESTATION=disabled bash Source/test_scripts/start_cloud.sh http
ATTESTATION=enabled bash Source/test_scripts/start_cloud.sh https-aes-gcm
ATTESTATION=enabled bash Source/test_scripts/start_cloud.sh https-chacha20
```

## 4. Run an Edge Experiment

`run_edge_comparison.sh` accepts three positional arguments:

```text
run_edge_comparison.sh <transport> <cloud-target> <inference-mode>
```

Supported transports:

- `http`
- `https-aes-gcm`
- `https-chacha20`

Supported inference modes:

- `split`: run split inference against a cloud VM;
- `local`: run the local inference baseline;
- `both`: run both split and local inference.

Example using the normal VM:

```bash
ATTESTATION=disabled bash Source/test_scripts/run_edge_comparison.sh \
  http 10.128.0.7 split
```

Example using the TDX VM:

```bash
ATTESTATION=enabled bash Source/test_scripts/run_edge_comparison.sh \
  https-aes-gcm 10.128.0.6 split
```

The input and batch size can be changed with environment variables:

```bash
IMAGE_DIR="$(pwd)/Source/datasets/ai2d" \
MAX_IMAGES=20 \
GENERATIONS=1 \
ATTESTATION=enabled \
bash Source/test_scripts/run_edge_comparison.sh \
  https-chacha20 10.128.0.6 split
```

## 5. Run the Complete Edge Comparison

After the cloud services are running on both VMs, execute:

```bash
bash Source/test_scripts/run_all_modes.sh
```

The script:

1. builds the TDX attestation helper;
2. runs HTTP and both TLS modes against the normal VM at `10.128.0.7`;
3. runs the same modes against the TDX VM at `10.128.0.6`;
4. runs the local inference baseline;
5. labels the generated records with the target address.

The two VM addresses are currently configured inside
`Source/test_scripts/run_all_modes.sh`.

## 6. Additional Checks

Check whether a cloud endpoint accepts TLS 1.3 or TLS 1.2:

```bash
bash Source/test_scripts/check_tls_version.sh 10.128.0.6 8443
```

Collect evidence about the normal VM environment:

```bash
bash Source/env/check_vm_environment.sh no_tdx Source/env/no_tdx
```

Collect evidence about the TDX VM environment:

```bash
bash Source/env/check_vm_environment.sh tdx Source/env/tdx
```

`test_attestation.py` and `test_transfer.py` contain unit tests for the
attestation and transfer components. `validate_split.py` compares generated
text from split inference with nanoVLM output.

## Output

All experiment scripts write their generated files to `Source/record/`:

- distributed_vlm_edge_*.log: edge execution logs.
- distributed_vlm_cloud_*.log: cloud execution logs.
- distributed_vlm_cloud_*.pid: cloud process IDs.
- distributed_vlm_metrics_*.csv: per-image metrics.
- distributed_vlm_throughput_*.csv: batch throughput metrics.

See the README in each subdirectory for more detailed information.
