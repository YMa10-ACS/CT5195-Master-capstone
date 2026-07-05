#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
import torch
import argparse
import psutil
from functools import wraps
from flask import Flask, request, jsonify
from vlm import init_tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from vlm import load_model, combined_image_and_text_message, construct_prompt, generate

import ssl

def time_using(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()

        result = func(*args, **kargs)

        end = time.perf_counter()
        print(f"[timeing] {func.__name__}: {end - start} seconds")

        return result
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--transport", choices=["http", "https-aes-gcm", "https-chacha20"], default="http",
                        help="Transport protocol used to send visual embeddings")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.device = device

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

app = Flask(__name__)

@app.route("/", methods=["POST"])
def receive_embedding():
    request_start = time.perf_counter()
    cpu_start = cpu_times()
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

    metrics = {
        "cloud_total_request_s": request_end - request_start,
        "cloud_cpu_total_s": cpu_end - cpu_start,
    }
    return jsonify({"result": text_list, "metrics": metrics})


if __name__ == "__main__":
    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_args()

    model = load_model(args.weights, args.device)
    tokenizer = init_tokenizer(model)

    app.config["prompt"] = args.prompt
    app.config["max_new_tokens"] = args.max_new_tokens
    app.config["device"] = args.device

    if args.transport == "http" :
        app.run(host=args.host, port=8000)
        exit

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # SSL context for HTTPS
    print(f"Starting HTTPS server on {args.host}:{args.transport}...")
    cert_path = os.path.join(PROJECT_ROOT, "cert", "cert.pem")
    key_path = os.path.join(PROJECT_ROOT, "cert", "key.pem")
    print(f"cert file: {cert_path}\n key file: {key_path}")

    context.load_cert_chain(cert_path, key_path)
    if args.transport == "https-aes-gcm":
        context.set_ciphers("ECDHE-RSA-AES128-GCM-SHA256")
        app.run(host=args.host, port=8443, ssl_context=context)
    elif args.transport == "https-chacha20":
        context.set_ciphers("ECDHE-RSA-CHACHA20-POLY1305")
        app.run(host=args.host, port=8444, ssl_context=context)

    exit
