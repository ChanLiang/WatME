# coding=utf-8
import os
import csv
import argparse
import numpy as np
import json
import torch
import tqdm
import time
import math
import pprint
import shortuuid
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from watermark_processor_syn_acl import WatermarkLogitsProcessor_with_synonym, WatermarkDetector_with_synonym
# from detect_repetition import remove_repeated_substrings

import datasets
# datasets.enable_caching()
from datasets import load_dataset
import re
from unidecode import unidecode

def shortest_repeated_substring(s):
    # Find all repeated substrings
    repeats = re.findall(r'(.+?)\1+', s)

    if repeats:
        # Sort the repeats by length, in ascending order
        repeats = sorted(repeats, key=len)

        # The shortest repeated substring is the first element
        shortest_repeat = repeats[0]

        # Replace all instances of the repeated substring with a single instance
        s = s.replace(shortest_repeat * 2, shortest_repeat)

    return s

def remove_repeated_substrings(s):
    while True:
        # Find and remove the shortest repeated substring
        new_s = shortest_repeated_substring(s)

        # If the string didn't change, we're done
        if new_s == s:
            break
        else:
            s = new_s

    return s

def remove_duplicate_sentences(text):
    sentences = text.split(". ")
    unique_sentences = list(dict.fromkeys(sentences))
    return ". ".join(unique_sentences)

def remove_repeated_or_substring_lines(text):
    lines = text.split('\n')
    output_lines = []

    for line in lines:
        line = unidecode(line)  # remove or replace special characters
        if len(output_lines) == 0 or (line != unidecode(output_lines[-1]) and line not in unidecode(output_lines[-1])):
            output_lines.append(line)

    return '\n'.join(output_lines)

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
        tokenizer.pad_token = "<PAD>"
        # tokenizer.pad_token = -1
        # tokenizer.bos_id = 1
        # tokenizer.eos_id = 2
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", device_map="auto", torch_dtype=torch.float16)
        # model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", device_map="auto", load_in_8bit=True)
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

def generate(turns, args, watermark_processor, tokenizer, model):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    device = 0
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )

    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    w_input, wo_input = [], []
    w_result, wo_result = [], []
    last_w_output, last_wo_output = '', ''
    for cur in range(len(turns)):
        turn = turns[cur]

        # w_prompt = f"User: {turn}\nSystem:"
        # wo_prompt = f"User: {turn}\nSystem:"
        w_prompt = f"Question: {turn}\nAnswer:"
        wo_prompt = f"Question: {turn}\nAnswer:"
        if cur > 0: # 有累计效应。
            # w_prompt = f"User: {turns[cur-1]}\nSystem: {last_w_output}\n" + w_prompt
            # wo_prompt = f"User: {turns[cur-1]}\nSystem: {last_wo_output}\n" + wo_prompt
            w_prompt = f"Question: {turns[cur-1]}\nAnswer: {last_w_output}\n" + w_prompt
            wo_prompt = f"Question: {turns[cur-1]}\nAnswer: {last_wo_output}\n" + wo_prompt

        w_input.append(w_prompt)
        wo_input.append(wo_prompt)
        
        w_tokd_input = tokenizer(w_prompt, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
        wo_tokd_input = tokenizer(wo_prompt, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)

        torch.manual_seed(args.generation_seed)
        output_without_watermark = generate_without_watermark(**wo_tokd_input)
        output_with_watermark = generate_with_watermark(**w_tokd_input)

        if args.is_decoder_only_model:
            # need to isolate the newly generated tokens
            output_without_watermark = output_without_watermark[:,wo_tokd_input["input_ids"].shape[-1]:]
            output_with_watermark = output_with_watermark[:,w_tokd_input["input_ids"].shape[-1]:]

        # print (f"len of wo watermark: {len(output_without_watermark[0])}")
        # print (f"len of w watermark: {len(output_with_watermark[0])}")

        # decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0].split('\n')[0] # generate multiple answers, but only take the first one
        # decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0].split('\n')[0]

        decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0].split('Question:')[0].strip() # generate multiple answers, but only take the first one
        decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0].split('Question:')[0].strip()

        if args.remove_output_repetition:
            decoded_output_without_watermark = remove_repeated_or_substring_lines(remove_repeated_substrings(decoded_output_without_watermark))
            decoded_output_with_watermark = remove_repeated_or_substring_lines(remove_repeated_substrings(decoded_output_with_watermark))

        last_w_output = decoded_output_with_watermark
        last_wo_output = decoded_output_without_watermark

        w_result.append(decoded_output_with_watermark)
        wo_result.append(decoded_output_without_watermark)

    assert len(w_result) == len(wo_result) == len(w_input) == len(wo_input) == 2, (len(w_result), len(wo_result), len(w_input), len(wo_input))

    return (
            wo_input,
            w_input,
            wo_result,
            w_result, 
            ) 

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

def read_mt_bench(path):
    with open(path, 'r', encoding='utf-8') as r: # jsonl
        res = []
        for i, line in enumerate(r):
            example = json.loads(line)
            res.append(example)
    return res

def main(args): 
    # load model and dataset
    # dataset_itr = load_exp_dataset(args) # realnewslike
    mt_bench_test = read_mt_bench('data/mt_bench/question.jsonl')
    args.num_samples = min(len(mt_bench_test), args.num_samples)
    model, tokenizer = load_model(args.model_name_or_path)

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

    file_path = f"output/{args.synonym_method}/mt_bench_completely_{args.model_name_or_path}_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_result" if args.use_synonyms else f"output/mt_bench_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_result"
    if args.remove_output_repetition:
        file_path = f"output/{args.synonym_method}/mt_bench_completely_{args.model_name_or_path}_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_removeRepetition_result" if args.use_synonyms else f"output/mt_bench_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_removeRepetition_result"
    file_path += f'_{args.max_new_tokens}.jsonl'

    f_output = open(file_path, 'w', encoding='utf-8')
    f_mt_eval_w = open(file_path + '_w_mt', 'w', encoding='utf-8')
    f_mt_eval_wo = open(file_path + '_wo_mt', 'w', encoding='utf-8')

    # generate and detect line by line. can we batch it?
    for id in tqdm.tqdm(range(args.num_samples)):
        cur_res = {}
        # input_text = [next(dataset_itr)['text']]
        test = mt_bench_test[id]
        turns = test['turns']
        assert len(turns) == 2, len(turns)

        # generate with and without watermark
        wo_inputs, w_inputs, decoded_output_without_watermarks, decoded_output_with_watermarks = generate(turns, args, watermark_processor, model=model, tokenizer=tokenizer)

        w_choices, wo_choices = [], []
        for cur in range(len(decoded_output_without_watermarks)): # treat each turn as a separate example
            decoded_output_with_watermark, decoded_output_without_watermark = decoded_output_with_watermarks[cur], decoded_output_without_watermarks[cur]
            wo_input, w_input = wo_inputs[cur], w_inputs[cur]

            # prefix = question
            cur_res["w_prompt"] = w_input
            cur_res["wo_prompt"] = wo_input
            cur_res["w_suffix"] = test['reference'][cur] if 'reference' in test else ''
            cur_res["wo_suffix"] = test['reference'][cur] if 'reference' in test else ''
            cur_res["wo_watermark_output"] = decoded_output_without_watermark
            cur_res["w_watermark_output"] = decoded_output_with_watermark

            if len(decoded_output_without_watermark.split()) > 1 and len(decoded_output_with_watermark.split()) > 1:
                # detect
                without_watermark_detection_result = detect(decoded_output_without_watermark, args, watermark_detector)
                with_watermark_detection_result = detect(decoded_output_with_watermark, args, watermark_detector)

                cur_res["wo_watermark_detection"] = without_watermark_detection_result[0]
                cur_res["w_watermark_detection"] = with_watermark_detection_result[0]

            f_output.write(json.dumps(cur_res) + '\n')
            f_output.flush()

            if args.debug_mode:
                term_width=50
                print("#"*term_width)
                print(f"Sample {id} of {args.num_samples}")

                print("#"*term_width)
                print ("Input:")
                print (wo_input)
                print("Output without watermark:")
                print(decoded_output_without_watermark)
                # print("-"*term_width)
                print(f"Detection result @ {args.detection_z_threshold}:")
                print(cur_res["wo_watermark_detection"])
                print("-"*term_width)

                print("#"*term_width)
                print("Input:")
                print (w_input)
                print("Output with watermark:")
                print(decoded_output_with_watermark)
                print("-"*term_width)
                print(f"Detection result @ {args.detection_z_threshold}:")
                print(cur_res["w_watermark_detection"])
                print("-"*term_width)

        w_choices.append({"index": 0, "turns": decoded_output_with_watermarks})
        wo_choices.append({"index": 0, "turns": decoded_output_without_watermarks})

        w_ans_json = {
            "question_id": test["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": 'llama2-7b',
            "choices": w_choices,
            "tstamp": time.time(),
        }
        wo_ans_json = {
            "question_id": test["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": 'llama2-7b',
            "choices": wo_choices,
            "tstamp": time.time(),
        }
        f_mt_eval_w.write(json.dumps(w_ans_json) + "\n")
        f_mt_eval_wo.write(json.dumps(wo_ans_json) + "\n")

    return

if __name__ == "__main__":

    args = parse_args()

    main(args)
