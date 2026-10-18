#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
import torch
from flask import Flask, request, jsonify
from vlm import init_tokenizer
from metrics import MetricsRecorder, names as metric_names
from attestation import AttestationError, create_evidence, is_attestation_enabled

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import nanoVLM.models.language_model as lm
from vlm import load_model, combined_image_and_text_message, construct_prompt, generate


projected_cache_key = None
projected_array = None
projected_cpu_tensor = None

# Read model and generation settings from environment variables and select the compute device.
def parse_env():
    args = dict()
    args["weights"] = os.environ.get("WEIGHTS", "lusxvr/nanoVLM-230M-8k")
    args["prompt"] = os.environ.get("PROMPT", "What is this?")
    args["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", "300"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args["device"] = device

    return args


# Reuse or allocate a NumPy embedding buffer and its reshaped CPU tensor for the requested shape and dtype.
def get_projected_buffer(shape, np_dtype):
    global projected_cache_key, projected_array, projected_cpu_tensor

    shape = tuple(shape)
    projected_shape = (1, -1, shape[-1])
    cache_key = (shape, np.dtype(np_dtype).str)

    if projected_cache_key != cache_key:
        projected_cache_key = cache_key
        projected_array = np.empty(shape, dtype=np_dtype)
        projected_cpu_tensor = torch.from_numpy(projected_array).reshape(projected_shape)
        app.logger.info(
            "allocated projected buffer:\n"
            f"  shape: {shape}\n"
            f"  dtype: {np.dtype(np_dtype)}"
        )

    return projected_array, projected_cpu_tensor


# Decode request metadata and binary embeddings, copy them into the reusable buffer, and select the target device.
def reconstruct_from_request(device):
    raw = request.data
    meta_json = request.headers.get("Visual-Meta")

    metadata = json.loads(meta_json)
    shape = metadata["shape"]
    dtype = metadata["dtype"]
    splitted_image_ratio = metadata["splitted_image_ratio"]
    generations = int(metadata["generations"])

    np_dtype = np.float32 if dtype == "float32" else np.float16
    raw_view = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    data, projected_cpu = get_projected_buffer(shape, np_dtype)
    np.copyto(data, raw_view)

    app.logger.info(f"data = {data.shape}")
    app.logger.info(f"data dtype: {data.dtype}")

    projected = projected_cpu if device == "cpu" else projected_cpu.to(device)
    return projected, splitted_image_ratio, generations

app = Flask(__name__)
model = None
tokenizer = None
prompt_cache = {}


# Return attestation evidence for the request, or an error when attestation is disabled or evidence creation fails.
@app.route("/attestation", methods=["POST"])
def attestation():
    if not is_attestation_enabled():
        app.logger.warning("Rejected attestation request because ATTESTATION=disabled")
        return jsonify({"error": "Attestation is disabled"}), 404

    app.logger.info("Received attestation request")
    try:
        evidence = create_evidence(request.get_json(silent=False))
        app.logger.info("Attestation evidence created successfully")
    except AttestationError as error:
        app.logger.error(f"Error creating attestation evidence: {error}")
        return jsonify({"error": str(error)}), 400
    return jsonify(evidence)


# Load and cache the cloud model and tokenizer when needed, and log initialization time.
def load_cloud_model(weights, device):
    global model, tokenizer

    if model is None or tokenizer is None:
        load_start = time.perf_counter()
        model = load_model(weights, device)
        tokenizer = init_tokenizer(model)
        load_end = time.perf_counter()
        app.logger.info(f"cloud model loaded in {load_end - load_start:.3f} seconds")

    return model, tokenizer


# Reuse cached prompt tokens or construct them for the requested device, prompt, and image tile layout.
def get_prompt_tokens(device, prompt, splitted_image_ratio):
    cache_key = (device, prompt, tuple(splitted_image_ratio))
    tokens = prompt_cache.get(cache_key)
    if tokens is None:
        tokens = construct_prompt(model, device, prompt, tokenizer, splitted_image_ratio)
        prompt_cache[cache_key] = tokens
    return tokens


# Replace image token positions in the prompt embeddings with the projected visual embeddings.
def combine_projected_with_prompt(tokens, projected):
    return combined_image_and_text_message(model, tokens, projected)


# Generate text from combined embeddings using greedy decoding and record generation metrics.
def generate_from_embedding(generations, device, tokens, token_embd, max_new_tokens, metrics):
    return generate(
                    model,
                    generations,
                    device=device,
                    input_ids=tokens,
                    token_embd=token_embd,
                    max_new_tokens=max_new_tokens,
                    greedy=True,
                    temperature=1.0,
                    timing_metrics=metrics
                )

# Reconstruct the visual embeddings and generate text while recording stage timings.
def process_embedding_request(metrics):
    device = app.config["device"]
    prompt = app.config["prompt"]
    max_new_tokens = app.config["max_new_tokens"]

    metrics.start_counter(metric_names.CLOUD_RECONSTRUCT_TIME)
    projected, splitted_image_ratio, generations = reconstruct_from_request(device)
    metrics.stop_counter(metric_names.CLOUD_RECONSTRUCT_TIME)

    app.logger.info(
        "receive_embedding parameters:\n"
        f"  device: {device}\n"
        f"  prompt: {prompt}\n"
        f"  project_shape: {tuple(projected.shape)}\n"
        f"  split_ratio: {splitted_image_ratio}\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  generations: {generations}"
    )

    metrics.start_counter(metric_names.CLOUD_PROMPT_TIME)
    tokens = get_prompt_tokens(device, prompt, splitted_image_ratio)
    metrics.stop_counter(metric_names.CLOUD_PROMPT_TIME)

    with torch.inference_mode():
        metrics.start_counter(metric_names.CLOUD_COMBINE_TIME)
        token_embd = combine_projected_with_prompt(tokens, projected)
        metrics.stop_counter(metric_names.CLOUD_COMBINE_TIME)

        metrics.start_counter(metric_names.CLOUD_GENERATE_TIME)
        text_list = generate_from_embedding(
            generations,
            device,
            tokens,
            token_embd,
            max_new_tokens,
            metrics=metrics,
        )
        metrics.stop_counter(metric_names.CLOUD_GENERATE_TIME)

    return text_list


# Finalize process CPU usage and total request timing.
def finalize_request_metrics(metrics):
    metrics.stop_cpu_counter(metric_names.CLOUD_CPU_TOTAL)
    metrics.stop_cpu_sampling(
        "cloud_request",
        metric_names.CLOUD_CPU_AVG_PCT,
        metric_names.CLOUD_CPU_MAX_PCT,
    )
    metrics.stop_counter(metric_names.CLOUD_TOTAL_REQUEST)


# Handle an embedding request, finalize metrics even on failure, and return generated text and metrics as JSON.
@app.route("/", methods=["POST"])
def receive_embedding():
    metrics = MetricsRecorder()
    metrics.start_counter(metric_names.CLOUD_TOTAL_REQUEST)
    metrics.start_cpu_counter(metric_names.CLOUD_CPU_TOTAL)
    metrics.start_cpu_sampling("cloud_request", interval=0.5)

    try:
        text_list = process_embedding_request(metrics)
    finally:
        finalize_request_metrics(metrics)

    return jsonify({"result": text_list, "metrics": metrics.row()})

# Initialize the cloud model and Flask configuration, then return the application.
def create_app():
    app_name = os.path.basename(__file__)
    app.logger.info(f"{app_name} startup...")
    args = parse_env()

    torch.manual_seed(0)
    load_cloud_model(args["weights"], args["device"])

    app.config["prompt"] = args["prompt"]
    app.config["max_new_tokens"] = args["max_new_tokens"]
    app.config["device"] = args["device"]

    return app

if __name__ == "__main__":
    app = create_app()
    app.logger.info("language_model path: {}".format(lm.__file__))
    app.run(host="0.0.0.0", port=8000, threaded=False)
