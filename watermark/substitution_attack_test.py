# coding=utf-8
import os
import re
import csv
import copy
import argparse
import numpy as np
import json
import torch
import tqdm
import random
import math
import pprint
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from watermark_processor_syn_acl import WatermarkLogitsProcessor_with_synonym, WatermarkDetector_with_synonym
from detect_repetition import remove_repeated_substrings
from sklearn.metrics import roc_auc_score

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
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )

    parser.add_argument(
        "--math",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--remove_output_repetition",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--synonym_gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )

    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )

    parser.add_argument(
        "--few_shot",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use_synonyms",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="gpt2_g0.5_d2.0_bl",
    )

    parser.add_argument(
        "--synonym_method",
        type=str,
    )

    parser.add_argument(
        "--synonym_clusters_path",
        type=str,
    )
     
    parser.add_argument(
        "--answer_path",
        type=str,
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        # default="facebook/opt-6.7b",
        default="gpt2",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--oracle_model_name",
        type=str,
        # default="facebook/opt-6.7b",
        default="gpt2-xl",
        # default="EleutherAI/gpt-j-6b",
    )

    parser.add_argument(
        "--replace_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--prompt_min_length",
        type=int,
        default=50,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--min_sample_length",
        type=int,
        default=250,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_sample_length",
        type=int,
        default=500,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        # default=200,
        default=100,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--suppress_eos",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )

    parser.add_argument(
        "--debug_mode",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--is_decoder_only_model",
        type=str2bool,
        default=True,
    )
    args = parser.parse_args()
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    return args

def load_model(model_name):
    if 'vicuna' in model_name:
        print ("loading vicuna-7b-v1.5-16k")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", padding_side='left')
        # tokenizer.pad_token = -1
        # tokenizer.bos_id = 1
        # tokenizer.eos_id = 2
        tokenizer.pad_token = "<PAD>"
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
    elif 'llama1-65b' in model_name:
        print ("loading llama")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", padding_side='left')
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", device_map="auto", torch_dtype=torch.float16)
    else:
        return ValueError('error!')
    # return model, tokenizer
    return tokenizer

def detect(input_text, args, watermark_detector):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def read_truthfulqa(path):
    with open(path, 'r', encoding='utf-8') as r:
        spamreader = csv.reader(r, delimiter=',')
        res = []
        for i, parts in enumerate(spamreader):
            if i == 0:
                continue
            assert len(parts) == 7, (i, len(parts))
            res.append(parts)
        return res

def load_gsm8k():
    # gsm8k = load_dataset('gsm8k', 'main')
    # gsm8k = load_dataset('gsm8k', cache_dir='/apdcephfs/share_1594716/chenliang/cache/gsm8k')
    # gsm8k = load_dataset('gsm8k', 'main', cache_dir='/apdcephfs/share_1594716/chenliang/cache/gsm8k')
    # gsm8k_test = gsm8k['test']
    # read jsonl
    gsm8k_test = []
    with open('data/gsm8k/test.jsonl', 'r') as f:
        for line in f:
            gsm8k_test.append(json.loads(line))
    return gsm8k_test

def filter_answer_line(output):
    lines = output.strip().split('\n')
    if lines[-1].lower().startswith("the answer is"):
        return '\n'.join(lines[:-1])
    else:
        return output

def read_responses(path):
    res = []
    with open(path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)

            # question = data['prompt'].split('Q: ')[-1].split('\nA:')[0]
            prompt = data['prompt'].strip()
            w_answer = data['w_watermark_output'].strip()
            wo_answer = data['wo_watermark_output'].strip()
            res.append([prompt, w_answer, wo_answer])
    return res

def replace_tokens_randomly(text, tokenizer, replace_ratio=0.1):
    # 对文本进行 tokenization
    tokenized = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized)

    # token_id_range = len(tokenizer.vocab)  # 获取 tokenizer 词汇表的大小
    token_id_range = len(tokenizer.get_vocab())
    
    num_tokens_to_replace = int(len(token_ids) * replace_ratio)

    # 随机替换 token ID
    for _ in range(num_tokens_to_replace):
        index_to_replace = random.randint(0, len(token_ids) - 1)

        # 使用当前 token ID 作为种子
        random.seed(token_ids[index_to_replace])
        
        # 生成一个随机 token ID
        random_token_id = random.randint(0, token_id_range - 1)

        # 替换 token
        token_ids[index_to_replace] = random_token_id

    # 将 token ID 转换回 tokens
    new_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # 将 tokens 重新组合成文本
    return tokenizer.convert_tokens_to_string(new_tokens)


def main(args): 
    gsm8k_test = load_gsm8k()
    tokenizer = load_model(args.model_name_or_path)

    prompt_answer_list = read_responses(args.answer_path)

    watermark_processor, watermark_detector = None, None
    if args.use_synonyms:
        watermark_processor = WatermarkLogitsProcessor_with_synonym(vocab=list(tokenizer.get_vocab().values()),
                                                        synonym_gamma=args.synonym_gamma, gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                                        synonym_clusters_path=args.synonym_clusters_path, math=args.math)

        watermark_detector = WatermarkDetector_with_synonym(vocab=list(tokenizer.get_vocab().values()), tokenizer=tokenizer, device='cuda:0',
                                            synonym_gamma=args.synonym_gamma, gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                            z_threshold=args.detection_z_threshold, normalizers=args.normalizers,
                                            synonym_clusters_path=args.synonym_clusters_path, math=args.math)
                                            
    else:
        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme)
    
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()), tokenizer=tokenizer, device='cuda:0', 
                                            gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                            z_threshold=args.detection_z_threshold, normalizers=args.normalizers)

    w_res, wo_res = [], []
    # for id in tqdm.tqdm(range(len(prompt_answer_list))):
    for id in tqdm.tqdm(range(200)):
    # for id in tqdm.tqdm(range(500)):

        # if len(prompt_answer_list) < 3:
            # continue 
        prompt, w_answer, wo_answer = prompt_answer_list[id]

        w_answer = replace_tokens_randomly(w_answer, tokenizer, args.replace_ratio)
        
        w_detect_res = detect(w_answer, args, watermark_detector)
        wo_detect_res = detect(wo_answer, args, watermark_detector)
        # print (w_detect_res[0][3][-1])
        w_res.append(w_detect_res[0][3][-1])
        wo_res.append(wo_detect_res[0][3][-1])

    positive_predictions, negative_predictions = w_res, wo_res
    auroc = roc_auc_score([1]*len(positive_predictions) + [0]*len(negative_predictions), positive_predictions + negative_predictions)
    print (auroc)
    
        
    return

if __name__ == "__main__":

    args = parse_args()

    main(args)
