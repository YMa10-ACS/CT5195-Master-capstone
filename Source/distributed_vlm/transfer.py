#!/usr/bin/env python
'''
Description:
Date: 2026-08-10 00:14:01
Author: Yaoquan Ma
'''

"""Transfer visual embeddings from the edge process to the cloud service."""

import json
import os
import time

import psutil
import requests

from attestation import (
    AttestationError,
    create_challenge,
    encode_challenge,
    verify_evidence,
)
from metrics import names as metric_names


# TransferError provides a unified abstraction for transport-related failures,
# including network errors, HTTP errors, and invalid response data.
class TransferError(RuntimeError):
    """Raised when an embedding transfer or cloud response is invalid."""


class EmbeddingTransferClient:
    """Reuse one HTTP session for all embedding transfers in a batch."""

    def __init__(self, target, transport_protocol):
        _target_url_for_transport(target, transport_protocol)
        self.target = target
        self.transport_protocol = transport_protocol
        self.session = requests.Session()
        self.attested = False

    def attest(self):
        """Request and verify fresh attestation evidence over this session."""
        nonce = create_challenge()
        challenge = encode_challenge(nonce)
        # Print the nonce in hex format for easier debugging and logging.
        print(f"Requesting attestation from {self.target} with nonce {nonce.hex()}")

        target_url = _target_url_for_transport(self.target, self.transport_protocol)
        attestation_url = f"{target_url}/attestation"
        request_kwargs = {"json": challenge}
        if self.transport_protocol != "http":
            request_kwargs["verify"] = False

        try:
            response = self.session.post(attestation_url, **request_kwargs)
            response.raise_for_status()
            evidence = response.json()

            # Print the evidence in JSON format for easier debugging and logging.
            print(f"Received attestation evidence: {json.dumps(evidence)}")
        except ValueError as error:
            raise AttestationError(
                "Cloud attestation response is not valid JSON"
            ) from error
        except requests.RequestException as error:
            raise AttestationError(
                f"Cloud attestation request failed at {attestation_url}: {error}"
            ) from error

        self.attested = verify_evidence(evidence, nonce)

        # Print the attestation result for easier debugging and logging.
        if self.attested:
            print(f"Attestation succeeded for {self.target}")
        else:
            print(f"Attestation failed for {self.target}")

        return True

    def transfer(self, projected, splitted_image_ratio, prompt, generations):
        if not self.attested:
            raise TransferError("Remote attestation must succeed before transfer")
        return transfer_embedding(
            self.session,
            self.target,
            self.transport_protocol,
            projected,
            splitted_image_ratio,
            prompt,
            generations,
        )

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _cpu_times():
    times = psutil.Process(os.getpid()).cpu_times()
    return times.user + times.system


def _target_url_for_transport(target, transport_protocol):
    if transport_protocol == "http":
        return f"http://{target}:8000"
    if transport_protocol == "https-aes-gcm":
        return f"https://{target}:8443"
    if transport_protocol == "https-chacha20":
        return f"https://{target}:8444"
    raise ValueError(f"Unsupported transport protocol: {transport_protocol}")


def _parse_response(response):
    try:
        response.raise_for_status()
        result_json = response.json()
    except ValueError as error:
        raise TransferError("Cloud response is not valid JSON") from error
    except requests.RequestException as error:
        raise TransferError(f"Cloud request failed: {error}") from error

    if not isinstance(result_json, dict):
        raise TransferError("Cloud response must be a JSON object")

    cloud_result = result_json.get("result")
    if not isinstance(cloud_result, list):
        raise TransferError("Cloud response is missing the result list")

    cloud_metrics = result_json.get("metrics")
    if not isinstance(cloud_metrics, dict):
        raise TransferError("Cloud response is missing the metrics object")
    if metric_names.CLOUD_TOTAL_REQUEST not in cloud_metrics:
        raise TransferError(
            f"Cloud metrics are missing {metric_names.CLOUD_TOTAL_REQUEST}"
        )

    return cloud_result, cloud_metrics

def transfer_embedding(
    session,
    target,
    transport_protocol,
    projected,
    splitted_image_ratio,
    prompt,
    generations,
):
    """Send one embedding using a caller-owned, reusable requests session."""

    data = projected.detach().cpu().numpy()
    metadata = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "prompt": str(prompt),
        "splitted_image_ratio": list(splitted_image_ratio),
        "generations": int(generations),
    }
    payload = data.tobytes()
    target_url = _target_url_for_transport(target, transport_protocol)

    request_kwargs = {
        "data": payload,
        "headers": {"Visual-Meta": json.dumps(metadata)},
    }
    if transport_protocol != "http":
        request_kwargs["verify"] = False

    transfer_start = time.perf_counter()
    transfer_cpu_start = _cpu_times()

    try:
        response = session.post(target_url, **request_kwargs)
    except requests.RequestException as error:
        raise TransferError(
            f"Embedding transfer failed for {transport_protocol} at {target_url}: {error}"
        ) from error

    transfer_cpu_end = _cpu_times()
    transfer_end = time.perf_counter()

    result_list, cloud_metrics = _parse_response(response)

    # Measure the network latency and CPU time for the transfer, and combine with cloud metrics.
    metrics = {
        metric_names.PAYLOAD_BYTES: len(payload),
        metric_names.TRANSFER_LATENCY: transfer_end - transfer_start,
    }

    metrics[metric_names.EDGE_TRANSFER_CPU_TOTAL] = (
        transfer_cpu_end - transfer_cpu_start
    )

    metrics.update(cloud_metrics)
    metrics[metric_names.NETWORK_OVERHEAD] = (
        metrics[metric_names.TRANSFER_LATENCY]
        - metrics[metric_names.CLOUD_TOTAL_REQUEST]
    )

    return result_list, metrics
