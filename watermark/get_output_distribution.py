# coding=utf-8
import os
import csv
import argparse
import numpy as np
import json
import torch
import tqdm
import math
import pprint
from functools import partial
from datasets import load_dataset
from demonstrations.truthfulqa_demonstrations import demonstrations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from watermark_processor_syn_acl import WatermarkLogitsProcessor_with_synonym, WatermarkDetector_with_synonym
from detect_repetition import remove_repeated_substrings
from scipy.stats import entropy

import datasets
# datasets.enable_caching()
from datasets import load_dataset

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")
    parser.add_argument(
        "--model_path",
        type=str,
        default="vicuna",
    )
    parser.add_argument(
        "--answer_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    args = parser.parse_args()
    return args

def load_model(model_name):
    if 'vicuna' in model_name:
        print ("loading vicuna-7b-v1.5-16k")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        # tokenizer.pad_token = -1
        # tokenizer.bos_id = 1
        # tokenizer.eos_id = 2
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", device_map="auto", torch_dtype=torch.float16)
        # model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-13b-v1.5-16k/", device_map="auto", load_in_8bit=True)
    elif 'llama2-chat-70b' in model_name:
        print ("loading llama2-chat-70b")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-70b-chat-hf", padding_side='left')
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = "<PAD>"
        # model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-70b-chat-hf", device_map="auto", torch_dtype=torch.float16)
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
    elif 'llama2-70b' in model_name:
        print ("loading llama2-70b")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-70b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
    elif 'llama2-chat-13b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-13b-chat-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-13b-chat-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.float16)
    elif 'llama2-13b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-13b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-13b-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.float16)
    elif 'llama2-7b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-7b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/Llama-2-7b-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.float16)
    elif 'llama1-65b' in model_name:
        print ("loading llama")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", padding_side='left')
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", device_map="auto", torch_dtype=torch.float16)
    else:
        return ValueError('error!')
    return model, tokenizer

# Calculate the average entropy for the tokens in the generated response
def calculate_average_entropy(model, tokenizer, prompt, gen_response):
    # Tokenize the combined prompt and generated response
    combined_input = prompt + gen_response
    inputs = tokenizer(combined_input, return_tensors='pt')

    # Get the number of tokens in the prompt
    num_prompt_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].size(1)

    with torch.no_grad():
        # Get model output for the combined input
        outputs = model(**inputs)
        logits = outputs.logits

    # Isolate logits for the generated response (exclude the prompt)
    response_logits = logits[:, num_prompt_tokens-1:-1]  # Shift by one to the left because the labels are shifted

    # Convert logits to probabilities
    probabilities = torch.softmax(response_logits, dim=-1)

    # Calculate the entropy for each token's probability distribution
    token_entropies = entropy(probabilities.cpu().numpy(), base=2, axis=2)

    # Calculate the average entropy for the generated response tokens
    average_entropy = token_entropies.mean(axis=1)
    return average_entropy.item()  # Return as a Python float

# Example usage:
# model_name = 'gpt-model-name'
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# prompt = "The quick brown fox"
# gen_response = "jumps over the lazy dog"
# avg_entropy = calculate_average_entropy(model, tokenizer, prompt, gen_response)
# print(avg_entropy)

# def read_truthfulqa(path):
#     with open(path, 'r', encoding='utf-8') as r:
#         spamreader = csv.reader(r, delimiter=',')
#         res = []
#         for i, parts in enumerate(spamreader):
#             if i == 0:
#                 continue
#             assert len(parts) == 7, (i, len(parts))
#             res.append(parts)
#         return res

def read_responses(path):
    res = []
    with open(path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)

            # question = data['prompt'].split('Q: ')[-1].split('\nA:')[0]
            prompt = data['prompt'].strip()
            answer = data['w_watermark_output'].strip()
            res.append([prompt, answer])
    return res

def main(args): 
    # load model and dataset
    # truthfulqa_test = read_truthfulqa('data/TruthfulQA.csv')
    prompt_answer_list = read_responses(args.answer_path)
    # assert len(truthfulqa_test) == len(prompt_answer_list), (len(truthfulqa_test), len(prompt_answer_list))
    # assert 817 == len(prompt_answer_list), len(prompt_answer_list)

    model, tokenizer = load_model(args.model_path)
    res_list = []
    for id in tqdm.tqdm(range(len(prompt_answer_list))):
        # print ('=='*20)
        prompt, gen_answer = prompt_answer_list[id]
        # print (gen_answer)
        cur_res = calculate_average_entropy(model, tokenizer, prompt, gen_answer)
        res_list.append(cur_res)
        # print (cur_res)
    
    with open(args.save_path, 'w') as w:
        w.write(json.dumps(res_list))
    print (sum(cur_res)/len(cur_res))
        
if __name__ == "__main__":

    args = parse_args()

    main(args)