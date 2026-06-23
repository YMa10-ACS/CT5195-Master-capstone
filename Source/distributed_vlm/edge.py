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
import json

import time
from functools import wraps

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from data.processors import get_tokenizer
from vlm import load_model, init_image_processor, init_tokenizer
from vlm import construct_prompt, combined_image_and_text_message, generate

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
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.device = device

    return args

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

@time_using
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
        end_vision = time.perf_counter()

        # Convert visual features from the vision encoder and map them into same embedding space.
        projected = model.MP(vision_out)
        end_projected = time.perf_counter()

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

@time_using
def transfer_embdding(projected, splitted_image_ratio, prompt, generations):
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
    response = requests.post(
        "http://127.0.0.1:8000", 
        data=payload,
        headers={"Visual-Meta": json.dumps(metadata)}
    )

    print("status code:", response.status_code)
    result_json = response.json()
    result_list = result_json["result"]
    for i, text in enumerate(result_list, start=1):
        print(f"  >> Generation {i}: {text}")
    
    return result_list

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

def main():
    app_name = os.path.basename(__file__)
    print(f"{app_name} startup...")
    args = parse_args()

    model = load_model(args.weights, args.device)
    image_processor = init_image_processor(model)
    text_tokonizer = init_tokenizer(model)

    result_list = []
    nanovlm_list = []
    
    processed_image, splitted_image_ratio = image_preprocessing(args.image, model, image_processor)
    projected = generate_embedding(processed_image, model, args.device)
    result_list = transfer_embdding(projected, splitted_image_ratio, args.prompt, args.generations)
    nanovlm_list = generate_text_by_embedding(model, args.device, projected, splitted_image_ratio, args.prompt, text_tokonizer, args.generations, args.max_new_tokens)


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


if __name__ == "__main__":
    main()
