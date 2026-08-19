# Secure Distributed VLM

This project implements secure split inference for a Vision-Language Model (VLM). Image processing is performed on the edge, while language generation runs locally or on a remote cloud VM.

The experiments compare normal and Intel TDX confidential VMs using HTTP and
TLS transport modes.

## System Workflow

1. The edge node processes an input image and generates a visual embedding.
2. The edge node optionally verifies the TDX environment through remote attestation.
3. The visual embedding is transferred to the cloud.
4. The cloud completes language generation and returns the result.
5. The system records latency, throughput and resource usage metrics.

## Directory Structure

- distributed_vlm/: distributed inference applications and supporting components.
- nanoVLM/: the external nanoVLM framework.
- configs/: TLS configuration files.
- datasets/: dataset download script and experiment images.
- analysis/: experiment analysis and plotting scripts.
- bechmarks/: normal VM and TDX VM benchmark results.
- env/: VM environment inspection results.
- secrets/: experimental TLS certificate and private key.


## Main Scripts

Run the following scripts from the `Source/` directory:

- distributed_vlm/init.sh: prepares the distributed VLM environment.
- distributed_vlm/start_cloud.sh: starts the cloud inference service.
- distributed_vlm/run_edge_comparison.sh: runs an edge inference experiment.
- distributed_vlm/run_all_modes.sh: runs the configured inference and transport modes.
- comparison.sh: compares split inference with the original nanoVLM output.
- validate_split.py: validates the output of split and non-split inference.
- datasets/download_hf_images.py: downloads experiment images.
- env/check_vm_environment.sh: collects VM environment information.
