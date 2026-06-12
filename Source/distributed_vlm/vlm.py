#!/usr/bin/env python
'''
Description: 
Date: 2026-06-08 00:15:01
Author: Yaoquan Ma
'''

import torch
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from models.vision_language_model import VisionLanguageModel
from models.utils import top_k_top_p_filtering
from data.processors import get_tokenizer, get_image_string, get_image_processor

def load_model(weights, device):
    torch.manual_seed(0)
    model = VisionLanguageModel.from_pretrained(weights)
    model.eval().to(device)

    return model

def init_tokenizer(model) :
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template
    )

    return tokenizer

def init_image_processor(model) :

    # Build image processor (same as generate.py style)
    resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
    print("resize to max side len :", resize_to_max_side_len)

    # Create an image processor object with specified parameters
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        resize_to_max_side_len
    )

    return image_processor


def construct_prompt(model, device, prompt, tokenizer, splitted_image_ratio):
    splitted_image_ratio = tuple(splitted_image_ratio)
    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt["input_ids"], dtype=torch.long).to(device)
    
    return tokens

def combined_image_and_text_message(model, tokens, projected):

    token_embd = model.decoder.token_embedding(tokens) # [B, T_prompt_text, D_lm]
    projected = projected.to(token_embd.dtype)
    token_embd = model._replace_img_tokens_with_embd(tokens, token_embd, projected)

    return token_embd

@torch.inference_mode()
def generate(model, generations, device, input_ids, token_embd, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        
        text_list = []
        for i in range(generations) :
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
            
            out_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"  >> Generation {i+1}: {out_text}")
            text_list.append(out_text)

        return text_list