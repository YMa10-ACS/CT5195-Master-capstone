#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
import torch
import argparse
from functools import wraps
from flask import Flask, request, jsonify

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from models.vision_language_model import VisionLanguageModel
from models.utils import top_k_top_p_filtering
from data.processors import get_tokenizer, get_image_string


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
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--weights", default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.device = device
    print("[cloud] device:", device)

    return args

@time_using
def load_model(model_weights, device):
    model = VisionLanguageModel.from_pretrained(model_weights)
    model.eval().to(device)

    torch.manual_seed(0)

    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )

    return model, tokenizer

def construct_prompt(splitted_image_ratio):
    splitted_image_ratio = tuple(splitted_image_ratio)
    prompt = app.config["prompt"]
    device = app.config["device"]
    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt["input_ids"], dtype=torch.long).to(device)


    return tokens

def combined_image_and_text_message(tokens, projected):

    token_embd = model.decoder.token_embedding(tokens) # [B, T_prompt_text, D_lm]
    projected = projected.to(token_embd.dtype)
    token_embd = model._replace_img_tokens_with_embd(tokens, token_embd, projected)

    return token_embd

def reconstruct_from_request(device):
    raw = request.data
    meta_json = request.headers.get("Visual-Meta")

    metadata = json.loads(meta_json)
    shape = metadata["shape"]
    dtype = metadata["dtype"]
    splitted_image_ratio = metadata["splitted_image_ratio"]

    np_dtype = np.float32 if dtype == "float32" else np.float16
    data = np.frombuffer(raw, dtype=np_dtype)

    data = data.reshape(shape)  # (13, 64, 576)
    print(f"data = {data.shape}")
    print(f"data dtype: {data.dtype}")

    # rebuild projected tensor -> (1, 832, 576)
    projected = torch.tensor(data, device=device).reshape(1, -1, shape[-1])
    return projected, splitted_image_ratio

@torch.inference_mode()
def generate(model, device, input_ids, token_embd, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0) # Or token_embd.size(0)
        
        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = model.decoder(
            token_embd,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not model.decoder.lm_use_tokens:
            current_logits = model.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = model.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = model.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not model.decoder.lm_use_tokens:
                current_logits = model.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if model.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == model.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = model.tokenizer.eos_token_id
        
        return generated_ids

app = Flask(__name__)

@app.route("/", methods=["POST"])
def receive_embedding():
    device = app.config["device"]
    generations = app.config["generations"]
    max_new_tokens = app.config["max_new_tokens"]

    projected, splitted_image_ratio = reconstruct_from_request(device)

    # prompt -> input_ids
    tokens = construct_prompt(splitted_image_ratio)

    token_embd = combined_image_and_text_message(tokens, projected)

    text_list = []
    for i in range(generations) :
        gen_ids = generate(
            model=model,
            device=device,
            input_ids=tokens,
            token_embd=token_embd,
            max_new_tokens=max_new_tokens,
            greedy=True,
            temperature=1.0
        )

        out_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        print(f"  >> Generation {i+1}: {out_text}")
        text_list.append(out_text)

    return jsonify({"result": text_list})

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model(args.weights, args.device)
    app.config["prompt"] = args.prompt
    app.config["max_new_tokens"] = args.max_new_tokens
    app.config["generations"] = args.generations
    app.config["device"] = args.device

    app.run(host=args.host, port=args.port)
