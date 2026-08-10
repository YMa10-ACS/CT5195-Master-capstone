"""Transfer visual embeddings from the edge process to the cloud service."""

import json
import os
import time

import psutil

from metrics import names as metric_names


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


def transfer_embedding(
    session,
    target,
    transport_protocol,
    projected,
    splitted_image_ratio,
    prompt,
    generations,
):
    """Serialize and send a projected visual embedding to the cloud service."""
    transfer_cpu_start = _cpu_times()

    data = projected.detach().cpu().numpy()
    print("data type:", type(data))
    print("data shape: ", data.shape)
    print("data dtype: ", data.dtype)
    metadata = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "prompt": str(prompt),
        "splitted_image_ratio": list(splitted_image_ratio),
        "generations": int(generations),
    }

    payload = data.tobytes()
    target_url = _target_url_for_transport(target, transport_protocol)
    print(f"target_url: {target_url}")

    transfer_start = time.perf_counter()
    request_kwargs = {
        "data": payload,
        "headers": {"Visual-Meta": json.dumps(metadata)},
    }
    if transport_protocol != "http":
        request_kwargs["verify"] = False
    response = session.post(target_url, **request_kwargs)
    transfer_end = time.perf_counter()

    print("status code:", response.status_code)
    result_json = response.json()
    print("cloud metrics keys:", sorted(result_json.get("metrics", {}).keys()))
    result_list = result_json["result"]
    for index, text in enumerate(result_list, start=1):
        print(f"  >> Generation {index}: {text}")

    metrics = {
        metric_names.PAYLOAD_BYTES: len(payload),
        metric_names.TRANSFER_LATENCY: transfer_end - transfer_start,
    }
    metrics.update(result_json.get("metrics", {}))
    metrics[metric_names.NETWORK_OVERHEAD] = (
        metrics[metric_names.TRANSFER_LATENCY]
        - metrics[metric_names.CLOUD_TOTAL_REQUEST]
    )
    transfer_cpu_end = _cpu_times()
    metrics[metric_names.EDGE_TRANSFER_CPU_TOTAL] = (
        transfer_cpu_end - transfer_cpu_start
    )
    return result_list, metrics
