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

from vlm import load_model, combined_image_and_text_message, construct_prompt, generate


def time_using(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()

        result = func(*args, **kargs)

        end = time.perf_counter()
        print(f"[timeing] {func.__name__}: {end - start} seconds")

        return result
    return wrapper

def parse_env():
    args = dict()
    args["weights"] = os.environ.get("WEIGHTS", "lusxvr/nanoVLM-230M-8k")
    args["prompt"] = os.environ.get("PROMPT", "What is this?")
    args["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", "300"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args["device"] = device

    return args


def reconstruct_from_request(device):
    raw = request.data
    meta_json = request.headers.get("Visual-Meta")

    metadata = json.loads(meta_json)
    shape = metadata["shape"]
    dtype = metadata["dtype"]
    splitted_image_ratio = metadata["splitted_image_ratio"]
    generations = int(metadata["generations"])

    np_dtype = np.float32 if dtype == "float32" else np.float16
    data = np.frombuffer(raw, dtype=np_dtype)

    data = data.reshape(shape)  # (13, 64, 576)
    print(f"data = {data.shape}")
    print(f"data dtype: {data.dtype}")

    # rebuild projected tensor -> (1, 832, 576)
    projected = torch.tensor(data, device=device).reshape(1, -1, shape[-1])
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

@app.route("/", methods=["POST"])
def receive_embedding():
    request_start = time.perf_counter()
    cpu_start = cpu_times()
    cpu_sampler = start_cpu_sampler(interval=0.5)
    torch.manual_seed(0)
    device = app.config["device"]
    prompt = app.config["prompt"]
    max_new_tokens = app.config["max_new_tokens"]

    projected, splitted_image_ratio, generations = reconstruct_from_request(device)
    print(
        "receive_embedding parameters:\n"
        f"  device: {device}\n"
        f"  prompt: {prompt}\n"
        f"  project_shape: {tuple(projected.shape)}\n"
        f"  split_ratio: {splitted_image_ratio}\n"
        f"  max_new_tokens: {max_new_tokens}"
        f"  generations: {generations}"
    )
    # prompt -> input_ids
    tokens = construct_prompt(model, device, prompt, tokenizer, splitted_image_ratio)

    token_embd = combined_image_and_text_message(model, tokens, projected)

    text_list = generate(
                    model,
                    generations,
                    device=device,
                    input_ids=tokens,
                    token_embd=token_embd,
                    max_new_tokens=max_new_tokens,
                    greedy=True,
                    temperature=1.0
                )
    request_end = time.perf_counter()
    cpu_end = cpu_times()
    cloud_cpu_avg_pct, cloud_cpu_max_pct = end_cpu_sampler(cpu_sampler)

    metrics = {
        "cloud_total_request": request_end - request_start,
        "cloud_cpu_total": cpu_end - cpu_start,
        "cloud_cpu_avg_pct": cloud_cpu_avg_pct,
        "cloud_cpu_max_pct": cloud_cpu_max_pct,
    }
    return jsonify({"result": text_list, "metrics": metrics})

model = None
tokenizer = None

def create_app():
    global model, tokenizer

    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_env()

    model = load_model(args["weights"], args["device"])
    tokenizer = init_tokenizer(model)

    app.config["prompt"] = args["prompt"]
    app.config["max_new_tokens"] = args["max_new_tokens"]
    app.config["device"] = args["device"]

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
