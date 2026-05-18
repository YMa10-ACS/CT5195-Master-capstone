#!/usr/bin/env python
'''
Description: 
Date: 2026-02-16 00:22:01
Author: Yaoquan Ma
'''

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, NANOVLM_PATH)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

import torch
import requests
import numpy as np

from PIL import Image
import argparse
import json

import time
from functools import wraps

def time_using(func):
    @wraps(func)
    def wrapper(*args, **kargs) :
        start = time.perf_counter()

        result = func(*args, **kargs)

        end = time.perf_counter()
        print(f"[timeing] {func.__name__}: {end - start} seconds")
        
        return result
    return wrapper

@time_using
def parameter_process():
    default_image = os.path.join(BASE_PATH, "img", "cat.jpg")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--image", default=default_image)
    args = parser.parse_args()

    # 1) Device selection
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # 2) Load model
    model_weights = args.weights
    image = args.image
    return device, model_weights, image


@time_using
def load_model(model_weights, device):
    # 1) Load pretrained model 
    model = VisionLanguageModel.from_pretrained(model_weights)
    model.eval().to(device)

    # 2) Build image processor (same as generate.py style)
    resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
    print("resize to max side len :", resize_to_max_side_len)

    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        resize_to_max_side_len
    )
    print("image processor : ", image_processor)

    return model, image_processor

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
    return processed_image

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
def transfer_embdding(projected):
    # 5) Transfer embedding
    data = projected.detach().to(torch.float16).cpu().numpy()
    print("data type:", type(data))
    print("data shape: ", data.shape)
    print("data dtype: ", data.dtype)
    metadata = {
        "shape" : list(data.shape),
        "dtype" : str(data.dtype)
    }

    payload = data.tobytes()
    requests.post(
        "http://127.0.0.1:8000", 
        data=payload,
        headers={"Visual-Meta": json.dumps(metadata)}
    )

def main():

    device, model_weights, image = parameter_process()

    model, image_processor = load_model(model_weights, device)

    processed_image = image_preprocessing(image, model, image_processor)

    projected = generate_embedding(processed_image, model, device)    

    transfer_embdding(projected)


if __name__ == "__main__":
    main()
