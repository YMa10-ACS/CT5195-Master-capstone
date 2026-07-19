#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
import torch
import psutil
from functools import wraps
from flask import Flask, request, jsonify
from vlm import init_tokenizer
from monitor import ProcessSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import nanoVLM.models.language_model as lm
from vlm import load_model, combined_image_and_text_message, construct_prompt, generate

print("language_model path:", lm.__file__)

projected_cache_key = None
projected_array = None
projected_cpu_tensor = None

def timed_stage(metric_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            start = time.perf_counter()
            result = func(*args, **kargs)
            end = time.perf_counter()
            elapsed = end - start

            metrics = kargs["metrics"]
            metrics[metric_name] = elapsed
            print(f"[timing] {metric_name}: {elapsed} seconds")

            return result
        return wrapper
    return decorator

def parse_env():
    args = dict()
    args["weights"] = os.environ.get("WEIGHTS", "lusxvr/nanoVLM-230M-8k")
    args["prompt"] = os.environ.get("PROMPT", "What is this?")
    args["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", "300"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args["device"] = device

    return args


def get_projected_buffer(shape, np_dtype):
    global projected_cache_key, projected_array, projected_cpu_tensor

    shape = tuple(shape)
    projected_shape = (1, -1, shape[-1])
    cache_key = (shape, np.dtype(np_dtype).str)

    if projected_cache_key != cache_key:
        projected_cache_key = cache_key
        projected_array = np.empty(shape, dtype=np_dtype)
        projected_cpu_tensor = torch.from_numpy(projected_array).reshape(projected_shape)
        print(
            "allocated projected buffer:\n"
            f"  shape: {shape}\n"
            f"  dtype: {np.dtype(np_dtype)}"
        )

    return projected_array, projected_cpu_tensor


@timed_stage("cloud_reconstruct_time")
def reconstruct_from_request(device, metrics):
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

    print(f"data = {data.shape}")
    print(f"data dtype: {data.dtype}")

    projected = projected_cpu if device == "cpu" else projected_cpu.to(device)
    return projected, splitted_image_ratio, generations

def cpu_times():
    times = psutil.Process(os.getpid()).cpu_times()
    return times.user + times.system

def start_cpu_sampler(interval=0.5):
    sampler = ProcessSampler(pid=os.getpid(), interval=interval)
    sampler.start()
    return sampler

def end_cpu_sampler(sampler):
    sampler.stop()
    cpu_samples = [cpu for cpu, _ in sampler.samples]
    cpu_avg_pct = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    cpu_max_pct = max(cpu_samples) if cpu_samples else 0.0
    return cpu_avg_pct, cpu_max_pct

app = Flask(__name__)
model = None
tokenizer = None
prompt_cache = {}


def load_cloud_model(weights, device):
    global model, tokenizer

    if model is None or tokenizer is None:
        load_start = time.perf_counter()
        model = load_model(weights, device)
        tokenizer = init_tokenizer(model)
        load_end = time.perf_counter()
        print(f"cloud model loaded in {load_end - load_start:.3f} seconds")

    return model, tokenizer


@timed_stage("cloud_prompt_time")
def get_prompt_tokens(device, prompt, splitted_image_ratio, metrics):
    cache_key = (device, prompt, tuple(splitted_image_ratio))
    tokens = prompt_cache.get(cache_key)
    if tokens is None:
        tokens = construct_prompt(model, device, prompt, tokenizer, splitted_image_ratio)
        prompt_cache[cache_key] = tokens
    return tokens


@timed_stage("cloud_combine_time")
def combine_projected_with_prompt(tokens, projected, metrics):
    return combined_image_and_text_message(model, tokens, projected)


@timed_stage("cloud_generate_time")
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

@app.route("/", methods=["POST"])
def receive_embedding():
    request_start = time.perf_counter()
    cpu_start = cpu_times()
    cpu_sampler = start_cpu_sampler(interval=0.5)
    metrics = {}
    device = app.config["device"]
    prompt = app.config["prompt"]
    max_new_tokens = app.config["max_new_tokens"]

    projected, splitted_image_ratio, generations = reconstruct_from_request(
        device, metrics=metrics
    )
    print(
        "receive_embedding parameters:\n"
        f"  device: {device}\n"
        f"  prompt: {prompt}\n"
        f"  project_shape: {tuple(projected.shape)}\n"
        f"  split_ratio: {splitted_image_ratio}\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  generations: {generations}"
    )
    tokens = get_prompt_tokens(
        device, prompt, splitted_image_ratio, metrics=metrics
    )

    with torch.inference_mode():
        token_embd = combine_projected_with_prompt(
            tokens, projected, metrics=metrics
        )
        text_list = generate_from_embedding(
            generations,
            device,
            tokens,
            token_embd,
            max_new_tokens,
            metrics=metrics,
        )
    request_end = time.perf_counter()
    cpu_end = cpu_times()
    cloud_cpu_avg_pct, cloud_cpu_max_pct = end_cpu_sampler(cpu_sampler)

    metrics.update({
        "cloud_total_request": request_end - request_start,
        "cloud_cpu_total": cpu_end - cpu_start,
        "cloud_cpu_avg_pct": cloud_cpu_avg_pct,
        "cloud_cpu_max_pct": cloud_cpu_max_pct,
    })
    return jsonify({"result": text_list, "metrics": metrics})

def create_app():
    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_env()

    torch.manual_seed(0)
    load_cloud_model(args["weights"], args["device"])

    app.config["prompt"] = args["prompt"]
    app.config["max_new_tokens"] = args["max_new_tokens"]
    app.config["device"] = args["device"]

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, threaded=False)
