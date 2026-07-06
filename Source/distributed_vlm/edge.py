#!/usr/bin/env python
'''
Description: 
Date: 2026-02-16 00:22:01
Author: Yaoquan Ma
'''


import torch
import requests

from PIL import Image
import argparse
import csv
import json
import psutil

import time
from datetime import datetime, timezone
from functools import wraps

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from data.processors import get_tokenizer
from monitor import read_process_cpu_rss, ProcessSampler
from vlm import load_model, init_image_processor, init_tokenizer
from vlm import construct_prompt, combined_image_and_text_message, generate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--split", type=str, default="both", choices=["both", "local", "split"],
                        help="Whether to run the model locally or split between edge and cloud")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--target", type=str, default="127.0.0.1",
                        help="Target IP for the request")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of input images for batch mode")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process in batch mode")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--transport", choices=["http", "https-aes-gcm", "https-chacha20"], default="http",
                        help="Transport protocol used to send visual embeddings")
    parser.add_argument("--ca_cert", default=f"{PROJECT_ROOT}/cert/cert.pem", type=str, 
                        help="Path to CA certificate for HTTPS requests")
    parser.add_argument("--metrics_csv", default=None, type=str,
                        help="Optional CSV file to append per-request metrics")
    parser.add_argument("--throughput_csv", default=None, type=str,
                        help="Optional CSV file to append batch-level throughput metrics")

    args = parser.parse_args()
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be provided")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.device = device

    return args

METRICS_COLUMNS = [
    "timestamp",
    "image",
    "transport",
    "payload_bytes",
    "split_latency",
    "edge_embedding_latency",
    "transfer_latency",
    "network_overhead",
    "local_baseline_latency",
    "cloud_total_request",
    "edge_embedding_cpu_total",
    "edge_transfer_cpu_total",
    "local_baseline_cpu_total",
    "edge_cpu_total",
    "cloud_cpu_total",
]

THROUGHPUT_COLUMNS = [
    "transport",
    "split",
    "target",
    "image_count",
    "batch_total_seconds",
    "throughput_images_per_minute",
]

def cpu_times():
    times = psutil.Process(os.getpid()).cpu_times()
    return times.user + times.system

def start_cpu_sampler(interval=0.5):
    pid = os.getpid()
    pre_stats = read_process_cpu_rss(pid)
    rss_before_mb = pre_stats[1] if pre_stats is not None else 0.0

    sampler = ProcessSampler(pid=pid, interval=interval)
    sampler.start()

    return sampler, rss_before_mb

def end_cpu_sampler(sampler, rss_before_mb):
    sampler.stop()
    sampler.join()
    cpu_total = sampler.cpu_total
    rss_after_mb = sampler.rss_max_mb
    rss_peak_mb = max(rss_before_mb, rss_after_mb)
    return cpu_total, rss_peak_mb

def write_metrics_csv(path, row):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METRICS_COLUMNS})

def write_throughput_csv(path, row):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THROUGHPUT_COLUMNS)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in THROUGHPUT_COLUMNS})

def target_url_for_transport(target, transport_protocol):
    if transport_protocol == "http":
        return f"http://{target}:8000"
    if transport_protocol == "https-aes-gcm":
        return f"https://{target}:8443"
    if transport_protocol == "https-chacha20":
        return f"https://{target}:8444"
    raise ValueError(f"Unsupported transport protocol: {transport_protocol}")

def collect_images(image_dir, max_images):
    image_paths = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.lower().endswith(".jpg"):
            image_paths.append(os.path.join(image_dir, file_name))
            if max_images is not None and len(image_paths) >= max_images:
                break

    return image_paths

def get_single_image_path(image_dir, image):
    if image_dir:
        return os.path.join(image_dir, image)
    return image

def time_using(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()

        result = func(*args, **kargs)

        end = time.perf_counter()
        print(f"[timeing] {func.__name__}: {end - start} seconds")

        return result
    return wrapper

@time_using
def image_preprocessing(image, model, image_processor):
    # 1) Some image is RGBA (additional channel is alpha). So, I need to convert the image into RGB format.
    img = Image.open(image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)

    # 2) Some configs insert a global image token; keep behaviour consistent with generate.py
    # 3) We also need tokenizer here just to check tokeniser here just to check the global_image_token existence.
    # 4) Because processed_image may still contain an extra global image tensor, even if the tokenizer does not support a global image token.
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )

    # 
    if (not hasattr(tokenizer, "global_image_token")
        and splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
        processed_image = processed_image[1:]
    return processed_image, splitted_image_ratio

def generate_embedding(processed_image, model, device):
    img_t = processed_image.to(device)
    print("img_t type:", type(img_t))
    print("img_t shape:", img_t.shape)
    print("img_t dtype:", img_t.dtype)
    print("img_t device:", img_t.device)
    print("img_t sample:", img_t.flatten()[:10])


    # 4) EDGE compute: vision encoder + modality projector
    with torch.no_grad():
        # Convert image to vision embedding.
        vision_out = model.vision_encoder(img_t)

        # Convert visual features from the vision encoder and map them into same embedding space.
        projected = model.MP(vision_out)

    print("vision_out type:", type(vision_out))
    print("vision_out shape:", vision_out.shape)
    print("vision_out dtype:", vision_out.dtype)
    print("vision_out device:", vision_out.device)
    # print("vision_out sample:", vision_out.flatten()[:10])
    
    print("projected type:", type(projected))
    print("projected shape:", projected.shape)
    print("projected dtype:", projected.dtype)
    print("projected device:", projected.device)

    return projected

def transfer_embdding(target, transport_protocol, ca_cert,projected, splitted_image_ratio, prompt, generations):
    transfer_cpu_start = cpu_times()

    # 5) Transfer embedding
    data = projected.detach().cpu().numpy()
    print("data type:", type(data))
    print("data shape: ", data.shape)
    print("data dtype: ", data.dtype)
    metadata = {
        "shape" : list(data.shape),
        "dtype" : str(data.dtype),
        "prompt": str(prompt),
        "splitted_image_ratio": list(splitted_image_ratio),
        "generations": int(generations)
    }

    payload = data.tobytes()

    target_url = target_url_for_transport(target, transport_protocol)
    print(f"target_url: {target_url}")
    transfer_start = time.perf_counter()
    if transport_protocol == "http":
        response = requests.post(
            target_url,
            data=payload,
            headers={"Visual-Meta": json.dumps(metadata)}
        )
    else:
        response = requests.post(
            target_url,
            data=payload,
            headers={"Visual-Meta": json.dumps(metadata)},
            verify=False
        )
    transfer_end = time.perf_counter()

    print("status code:", response.status_code)
    result_json = response.json()
    result_list = result_json["result"]
    for i, text in enumerate(result_list, start=1):
        print(f"  >> Generation {i}: {text}")

    metrics = {
        "payload_bytes": len(payload),
        "transfer_latency": transfer_end - transfer_start,
    }
    metrics.update(result_json.get("metrics", {}))
    metrics["network_overhead"] = metrics["transfer_latency"] - metrics["cloud_total_request"]
    transfer_cpu_end = cpu_times()
    metrics["edge_transfer_cpu_total"] = transfer_cpu_end - transfer_cpu_start
    return result_list, metrics

def generate_text_by_embedding(model, device, projected, splitted_image_ratio, prompt, text_tokonizer, generations, max_new_tokens) :
    
    # Print generate_text_by_embedding parameters for comparison with the cloud service.
    print(
        "generate_text_by_embedding parameters:\n"
        f"  device: {device}\n"
        f"  prompt: {prompt}\n"
        f"  project_shape: {tuple(projected.shape)}\n"
        f"  split_ratio: {splitted_image_ratio}\n"
        f"  max_new_tokens: {max_new_tokens}"
        f"  generations: {generations}"
    )


    tokens = construct_prompt(model, device, prompt, text_tokonizer, splitted_image_ratio)

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
    return text_list

def process_image(args, image_path, model, image_processor, text_tokonizer):
    result_list = []
    nanovlm_list = []
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": image_path,
        "transport": args.transport,
    }

    # Start CPU sampler to measure CPU usage during embedding generation
    edge_embedding_cpu_sampler, rss_before_mb = start_cpu_sampler(interval=0.5)

    # Start CPU timer for the entire edge processing.
    # Calculate the total CPU time used by the edge device for this image processing.
    edge_cpu_start = cpu_times()

    split_start = time.perf_counter()
    embedding_cpu_start = cpu_times()
    processed_image, splitted_image_ratio = image_preprocessing(image_path, model, image_processor)

    # Start recording the time taken to generate the embedding for the processed image.
    embedding_start = time.perf_counter()
    projected = generate_embedding(processed_image, model, args.device)
    embedding_end = time.perf_counter()
    embedding_cpu_end = cpu_times()
    print(f"[timeing] generate_embedding: {embedding_end - embedding_start} seconds")
    metrics["edge_embedding_latency"] = embedding_end - embedding_start
    metrics["edge_embedding_cpu_total"] = embedding_cpu_end - embedding_cpu_start

    edge_embedding_cpu, _ = end_cpu_sampler(edge_embedding_cpu_sampler, rss_before_mb)
    metrics["edge_cpu_total"] = edge_embedding_cpu

    if args.split == "split" or args.split == "both":
        result_list, transfer_metrics = transfer_embdding(args.target, args.transport, args.ca_cert, projected, splitted_image_ratio, args.prompt, args.generations)
        split_end = time.perf_counter()
        metrics["split_latency"] = split_end - split_start
        metrics.update(transfer_metrics)
    

    if args.split == "local" or args.split == "both":
        local_generate_start = time.perf_counter()
        local_generate_cpu_start = cpu_times()
        nanovlm_list = generate_text_by_embedding(model, args.device, projected, splitted_image_ratio, args.prompt, text_tokonizer, args.generations, args.max_new_tokens)
        local_generate_cpu_end = cpu_times()
        local_generate_end = time.perf_counter()
        metrics["local_baseline_latency"] = local_generate_end - local_generate_start
        metrics["local_baseline_cpu_total"] = local_generate_cpu_end - local_generate_cpu_start
    
    edge_cpu_end = cpu_times()
    metrics["edge_cpu_total"] = edge_cpu_end - edge_cpu_start
    write_metrics_csv(args.metrics_csv, metrics)

    if args.split != "both":
        return

    unmatched_pair = 0
    matched_pair = 0
    for s1, s2 in zip(result_list, nanovlm_list) :
        if s1 != s2 :
            print(f"split leanring generate text : {s1}")
            print(f"nanoVLM generate text: {s2}")
            unmatched_pair += 1
        else :
            matched_pair += 1

    print(f"unmatch pair / matched pair = {unmatched_pair}/{matched_pair}")

def main():
    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_args()

    model = load_model(args.weights, args.device)
    image_processor = init_image_processor(model)
    text_tokonizer = init_tokenizer(model)

    if args.image:
        images = [get_single_image_path(args.image_dir, args.image)]
    else:
        images = collect_images(args.image_dir, args.max_images)
        print(f"collect {len(images)} images from {args.image_dir}")

    batch_start = time.perf_counter()
    for index, image_path in enumerate(images, start=1):
        print(f"===== Image {index}/{len(images)}: {os.path.basename(image_path)} =====")
        process_image(args, image_path, model, image_processor, text_tokonizer)
    batch_end = time.perf_counter()

    batch_total = batch_end - batch_start
    throughput_images_per_min = len(images) / batch_total * 60 if batch_total > 0 else 0
    print(f"throughput images: {len(images)}")
    print(f"throughput total seconds: {batch_total}")
    print(f"throughput images per minute: {throughput_images_per_min}")
    write_throughput_csv(args.throughput_csv, {
        "transport": args.transport,
        "split": args.split,
        "target": args.target,
        "image_count": len(images),
        "batch_total_seconds": batch_total,
        "throughput_images_per_minute": throughput_images_per_min,
    })

if __name__ == "__main__":
    main()
