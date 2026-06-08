#!/usr/bin/env python3
import argparse
import difflib
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
NANOVLM_PATH = os.path.join(ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from data.processors import get_tokenizer
from models.vision_language_model import VisionLanguageModel


GENERATION_PATTERN = re.compile(r"Generation\s+(\d+):\s*(.*)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-log", default="/tmp/distributed_vlm_edge.log")
    parser.add_argument("--nanovlm-log", default="/tmp/nanovlm_generate.log")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--image", default="")
    return parser.parse_args()


def read_generations(path):
    generations = []

    if not os.path.exists(path):
        return generations

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = GENERATION_PATTERN.search(line)
            if match:
                generations.append(match.group(2).strip())

    return generations


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text_embedding_model(weights, device):
    model = VisionLanguageModel.from_pretrained(weights).eval().to(device)
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template,
    )
    return model, tokenizer


@torch.inference_mode()
def text_embedding(text, model, tokenizer, device):
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    emb = model.decoder.token_embedding(input_ids)
    return emb.mean(dim=1).squeeze(0).float()


def embedding_distance(edge_text, nanovlm_text, model, tokenizer, device):
    edge_emb = text_embedding(edge_text, model, tokenizer, device)
    nanovlm_emb = text_embedding(nanovlm_text, model, tokenizer, device)
    diff = edge_emb - nanovlm_emb

    return {
        "cosine": F.cosine_similarity(edge_emb.unsqueeze(0), nanovlm_emb.unsqueeze(0)).item(),
}


def main():
    args = parse_args()

    edge_outputs = read_generations(args.edge_log)
    nanovlm_outputs = read_generations(args.nanovlm_log)

    print("===== Validation Result =====")
    if args.image:
        print("image:", args.image)
    print("edge outputs:", len(edge_outputs))
    print("nanoVLM outputs:", len(nanovlm_outputs))

    if not edge_outputs:
        print(f"No generation output found in edge log: {args.edge_log}")
        return

    if not nanovlm_outputs:
        print(f"No generation output found in nanoVLM log: {args.nanovlm_log}")
        return

    device = get_device()
    print("Using device:", device)
    model, tokenizer = load_text_embedding_model(args.weights, device)

    pair_count = min(len(edge_outputs), len(nanovlm_outputs))
    exact_matches = 0
    semantic_matches = 0

    for index in range(pair_count):
        edge_text = edge_outputs[index]
        nanovlm_text = nanovlm_outputs[index]
        is_match = edge_text == nanovlm_text
        distance = embedding_distance(edge_text, nanovlm_text, model, tokenizer, device)
        semantic_match = distance["cosine"] >= args.threshold

        if is_match:
            exact_matches += 1
        if semantic_match:
            semantic_matches += 1

        print(f"\nGeneration {index + 1}")
        print("edge:   ", edge_text)
        print("nanoVLM:", nanovlm_text)
        print("exact match:", is_match)
        print(f"embedding cosine: {distance['cosine']:.4f}")
        print(f"semantic match by threshold {args.threshold}: {semantic_match}")

    print("\n===== Summary =====")
    print(f"compared pairs: {pair_count}")
    print(f"exact matches: {exact_matches}/{pair_count}")
    print(f"semantic matches: {semantic_matches}/{pair_count}")
    


if __name__ == "__main__":
    main()
