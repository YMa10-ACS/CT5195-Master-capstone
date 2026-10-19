#!/usr/bin/env python
'''
Description: 
Date: 2026-02-16 00:22:01
Author: Yaoquan Ma
'''


import torch

from PIL import Image
import argparse

import time
from datetime import datetime, timezone
from functools import wraps
from contextlib import nullcontext

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from vlm import load_model, init_image_processor, init_tokenizer
from vlm import construct_prompt, combined_image_and_text_message, generate
from metrics import names as metric_names
from metrics import MetricsRecorder, write_metrics_csv, write_throughput_csv
from attestation import is_attestation_enabled
from transfer import EmbeddingTransferClient, TransferError

# Parse command-line options, validate image input, and select the compute device.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="both", choices=["both", "local", "split"],
                        help="Whether to run the model locally or split between edge and cloud")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input single image")
    parser.add_argument("--target", type=str, default="127.0.0.1",
                        help="Target IP for the request")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of input images for batch mode")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process in batch mode")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt for local inference; cloud inference uses the cloud prompt configuration")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--transport", choices=["http", "https-aes-gcm", "https-chacha20"], default="http",
                        help="Transport protocol used to send visual embeddings")
    parser.add_argument("--ca_cert", default=f"{PROJECT_ROOT}/secrets/cert.pem", type=str, 
                        help="Path to CA certificate for HTTPS requests")
    parser.add_argument("--metrics_csv", default=None, type=str,
                        help="Optional CSV file to append per-request metrics")
    parser.add_argument("--throughput_csv", default=None, type=str,
                        help="Optional CSV file to append batch-level throughput metrics")

    args = parser.parse_args()
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be provided")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    args.device = device

    return args

# Collect sorted JPG image paths from a directory, up to the optional limit.
def collect_images(image_dir, max_images):
    image_paths = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.lower().endswith(".jpg"):
            image_paths.append(os.path.join(image_dir, file_name))
            if max_images is not None and len(image_paths) >= max_images:
                break

    return image_paths

# Resolve a single image path using the optional image directory.
def get_single_image_path(image_dir, image):
    if image_dir:
        return os.path.join(image_dir, image)
    return image

# Decorate a function to print its elapsed execution time.
def time_using(func):
    # Run the wrapped function and print its elapsed time before returning its result.
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()

        result = func(*args, **kargs)

        end = time.perf_counter()
        print(f"[timeing] {func.__name__}: {end - start} seconds")

        return result
    return wrapper

# Load an RGB image, prepare image tensors, and remove an unsupported global image.
@time_using
def image_preprocessing(image, text_tokenizer, image_processor):
    # 1) Some image is RGBA (additional channel is alpha). So, I need to convert the image into RGB format.
    img = Image.open(image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)

    # 2) Some configs insert a global image token; keep behaviour consistent with generate.py
    # 3) Reuse the existing tokenizer to check for the global image token.
    # 4) Because processed_image may still contain an extra global image tensor, even if the tokenizer does not support a global image token.
    # 
    if (not hasattr(text_tokenizer, "global_image_token")
        and splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
        processed_image = processed_image[1:]
    return processed_image, splitted_image_ratio

# Encode image tensors and project visual features into the language model embedding space.
def generate_embedding(processed_image, model, device):
    img_t = processed_image.to(device)
    print("img_t type:", type(img_t))
    print("img_t shape:", img_t.shape)
    print("img_t dtype:", img_t.dtype)
    print("img_t device:", img_t.device)
    print("img_t sample:", img_t.flatten()[:10])


    # 4) EDGE compute: vision encoder + modality projector
    with torch.inference_mode():
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

# Combine the prompt with visual embeddings and generate text locally.
def generate_text_by_embedding(model, device, projected, splitted_image_ratio, prompt, text_tokenizer, generations, max_new_tokens) :
    
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


    tokens = construct_prompt(model, device, prompt, text_tokenizer, splitted_image_ratio)

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

# Compare split and local outputs, print differences, and count matching pairs.
def compare_generated_texts(result_list, nanovlm_list):
    if len(result_list) != len(nanovlm_list):
        raise ValueError("两个列表的长度必须一致")

    unmatched_pair = 0
    matched_pair = 0

    for s1, s2 in zip(result_list, nanovlm_list):
        if s1 != s2:
            print(f"split learning generate text: {s1}")
            print(f"nanoVLM generate text: {s2}")
            unmatched_pair += 1
        else:
            matched_pair += 1

    print(f"unmatched pair / matched pair = {unmatched_pair}/{matched_pair}")

# Decorate embedding generation to record elapsed time, CPU time, and sampled CPU usage.
def measure_embedding(func):
    # Run embedding generation and finalize its performance metrics even if it raises an error.
    @wraps(func)
    def wrapper(*args, metrics, **kwargs):
        metrics.start_cpu_sampling("embedding", interval=0.5)
        metrics.start_cpu_counter(metric_names.EDGE_EMBEDDING_CPU_TOTAL)
        metrics.start_counter(metric_names.EDGE_EMBEDDING_LATENCY)

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = metrics.stop_counter(metric_names.EDGE_EMBEDDING_LATENCY)
            metrics.stop_cpu_counter(metric_names.EDGE_EMBEDDING_CPU_TOTAL)
            metrics.stop_cpu_sampling(
                "embedding",
                metric_names.EDGE_EMBEDDING_CPU_AVG_PCT,
                metric_names.EDGE_EMBEDDING_CPU_MAX_PCT,
            )

            print(f"[timing] generate_embedding: {elapsed:.6f} seconds")

    return wrapper

# Process one image in the selected inference modes, save metrics, and compare outputs when using both modes.
def process_image(args, transfer_client, image_path, model, image_processor, text_tokenizer):
    result_list = []
    nanovlm_list = []
    metrics = MetricsRecorder({
        metric_names.TIMESTAMP: datetime.now(timezone.utc).isoformat(),
        metric_names.IMAGE: image_path,
        metric_names.TRANSPORT: args.transport,
    })

    processed_image, splitted_image_ratio = image_preprocessing(image_path, text_tokenizer, image_processor)

    projected = measure_embedding(generate_embedding)(
        processed_image, model, args.device, metrics=metrics
    )

    metrics.start_counter(metric_names.SPLIT_LATENCY)
    if args.split == "split" or args.split == "both":
        try:
            result_list, transfer_metrics = transfer_client.transfer(
                projected,
                splitted_image_ratio,
                args.generations,
            )
        except TransferError as error:
            raise RuntimeError(
                f"Unable to complete {args.transport} transfer for {image_path}: {error}"
            ) from error
        metrics.stop_counter(metric_names.SPLIT_LATENCY)

        metrics.update(transfer_metrics)
    

    if args.split == "local" or args.split == "both":
        
        metrics.start_cpu_counter(metric_names.LOCAL_BASELINE_CPU_TOTAL)
        metrics.start_counter(metric_names.LOCAL_BASELINE_LATENCY)
        nanovlm_list = generate_text_by_embedding(model, args.device, projected, splitted_image_ratio, args.prompt, text_tokenizer, args.generations, args.max_new_tokens)
        metrics.stop_counter(metric_names.LOCAL_BASELINE_LATENCY)
        metrics.stop_cpu_counter(metric_names.LOCAL_BASELINE_CPU_TOTAL)

    write_metrics_csv(args.metrics_csv, metrics.row())

    if args.split != "both":
        return

    compare_generated_texts(result_list, nanovlm_list)

# Create the transfer client context, or an empty context for local inference.
def init_transfer_client(args):
    if args.split == "local":
        return nullcontext(None)
    return EmbeddingTransferClient(
        args.target,
        args.transport,
        args.ca_cert,
    )

# Manage the client context, perform optional attestation, and return the image batch processing time.
def process_images(args, client_context, images, model, image_processor, text_tokenizer):
    with client_context as transfer_client:
        if args.split != "local" and is_attestation_enabled():
            transfer_client.attest()
        
        # The timer measures the total time spent processing the images, 
        # including preprocessing, inference, and data transfer. 
        # It excludes model loading, image path collection, client setup, and attestation.

        batch_start = time.perf_counter()
        for index, image_path in enumerate(images, start=1):
            print(f"===== Image {index}/{len(images)}: {os.path.basename(image_path)} =====")
            process_image(args, transfer_client, image_path, model, image_processor, text_tokenizer)
        batch_end = time.perf_counter()

    return batch_end - batch_start

# Calculate images per minute, print batch statistics, and write throughput metrics to CSV.
def report_throughput(args, images, batch_total):
    throughput_images_per_min = len(images) / batch_total * 60
    print(f"throughput images: {len(images)}")
    print(f"throughput total seconds: {batch_total}")
    print(f"throughput images per minute: {throughput_images_per_min}")
    write_throughput_csv(args.throughput_csv, {
        metric_names.TRANSPORT: args.transport,
        metric_names.SPLIT: args.split,
        metric_names.TARGET: args.target,
        metric_names.IMAGE_COUNT: len(images),
        metric_names.BATCH_TOTAL_SECONDS: batch_total,
        metric_names.THROUGHPUT_IMAGES_PER_MINUTE: throughput_images_per_min,
    })

# This is the main entry point of the edge application. 
# It handles initialization, input selection, client setup, image processing, and throughput reporting.

def main():
    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_args()

    model = load_model(args.weights, args.device)
    image_processor = init_image_processor(model)
    text_tokenizer = init_tokenizer(model)

    if args.image:
        images = [get_single_image_path(args.image_dir, args.image)]
    else:
        images = collect_images(args.image_dir, args.max_images)
        print(f"collect {len(images)} images from {args.image_dir}")

    client_context = init_transfer_client(args)

    batch_total = process_images(
        args, client_context, images, model, image_processor, text_tokenizer
    )
    
    report_throughput(args, images, batch_total)

if __name__ == "__main__":
    main()
