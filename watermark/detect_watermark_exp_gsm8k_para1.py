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
import math
import pprint
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from watermark_processor_syn_acl import WatermarkLogitsProcessor_with_synonym, WatermarkDetector_with_synonym
from detect_repetition import remove_repeated_substrings
from prompt_template import PromptTemplate as PT

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
        default=False,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )

    # fix_seed
    parser.add_argument(
        "--fix_seed",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--filter_answer_line_when_detecting",
        type=str2bool,
        default=False,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )

    parser.add_argument(
        "--remove_output_repetition",
        type=str2bool,
        default=False
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
        "--paraphraser",
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
        "--synonym_gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
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

def load_model(model_name, model=False):
    model = None
    if 'vicuna' in model_name:
        print ("loading vicuna-7b-v1.5-16k")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/vicuna-7b-v1.5-16k/", padding_side='left')
        # tokenizer.pad_token = -1
        # tokenizer.bos_id = 1
        # tokenizer.eos_id = 2
        tokenizer.pad_token = "<PAD>"
        model = LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/vicuna-7b-v1.5-16k/", device_map="auto", torch_dtype=torch.bfloat16)
    elif 'llama2-chat-70b' in model_name:
        print ("loading llama2-chat-70b")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-70b-chat-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
    elif 'llama2-70b' in model_name:
        print ("loading llama2-70b")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-70b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-70b-hf", device_map="auto", load_in_8bit=True)
    elif 'llama2-chat-13b' in model_name:
        print ("loading llama2-chat-13b")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-13b-chat-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-13b-chat-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)
    elif 'llama2-13b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-13b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-13b-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)
    elif 'llama2-7b' in model_name:
        print ("loading llama2")
        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-7b-hf", padding_side='left')
        tokenizer.pad_token = "<PAD>"
        model = AutoModelForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/Llama-2-7b-hf", use_auth_token=True, device_map="auto", torch_dtype=torch.bfloat16)
    elif 'llama1-65b' in model_name:
        print ("loading llama")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/65B/", padding_side='left')
        model = LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_1594716/willllchen/cache/65B/", device_map="auto", torch_dtype=torch.bfloat16)
    else:
        return ValueError('error!')
    return model, tokenizer
    # return tokenizer

def generate(prompt, args, watermark_processor, model, tokenizer):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    device=0
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_p=1.0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    # if args.suppress_eos:
    #     gen_kwargs.update(dict(suppress_tokens=[tokenizer.eos_token_id]))

    if 'gpt' in args.model_name_or_path: # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))

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

    # get the prefix and suffix
    # tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_min_length).to(device)
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
    prefix = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    # print (f"prefix length: {tokd_input['input_ids'].shape[-1]}")

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    # print (f"len of wo watermark: {len(output_without_watermark[0])}")
    # print (f"len of w watermark: {len(output_with_watermark[0])}")

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_without_watermark = decoded_output_without_watermark.split('Question')[0] # generate multiple answers, but only take the first one
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    # print (decoded_output_with_watermark)
    decoded_output_with_watermark = decoded_output_with_watermark.split('Question')[0]

    return (
            decoded_output_without_watermark.strip(), 
            decoded_output_with_watermark.strip(),
            ) 

def paraphrase_response(prompt, input_text, args, model, tokenizer):
    device=0

    input_text_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)['input_ids']
    # tokd_input = tokenizer(prompt.strip() + '\n' + input_text.strip(), return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
    tokd_input = tokenizer(prompt.strip(), return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
    prefix = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    # gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    # print('input_text_ids.shape[-1]  = ', input_text_ids.shape[-1])
    gen_kwargs = dict(max_new_tokens=input_text_ids.shape[-1] + 10)

    gen_kwargs.update(dict(
        num_beams=1
    ))

    if 'gpt' in args.model_name_or_path: # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))

    generate = partial(
        model.generate,
        **gen_kwargs
    )

    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    torch.manual_seed(args.generation_seed)
    output = generate(**tokd_input)

    if args.is_decoder_only_model:
        output = output[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # print('decoded_output = ', decoded_output)
    decoded_output = decoded_output.strip().split('\n\n')[1] # generate multiple answers, but only take the first one

    return decoded_output.strip()

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

def main(args): 
    # load model and dataset
    prompt = open("demonstrations/gsm8k_demonstrations.txt", 'r').read().strip()
    gsm8k_test = load_gsm8k()
    args.num_samples = min(len(gsm8k_test), args.num_samples)

    paraphraser, tokenizer_paraphraser = load_model(args.paraphraser)
    model, tokenizer = load_model(args.model_name_or_path)

    watermark_processor, watermark_detector = None, None
    if args.use_synonyms:
        watermark_processor = WatermarkLogitsProcessor_with_synonym(vocab=list(tokenizer.get_vocab().values()),
                                                        synonym_gamma=args.synonym_gamma, gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                                        synonym_clusters_path=args.synonym_clusters_path, math=args.math, fix_seed=args.fix_seed)

        watermark_detector = WatermarkDetector_with_synonym(vocab=list(tokenizer.get_vocab().values()), tokenizer=tokenizer, device='cuda:0',
                                            synonym_gamma=args.synonym_gamma, gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                            z_threshold=args.detection_z_threshold, normalizers=args.normalizers,
                                            synonym_clusters_path=args.synonym_clusters_path, math=args.math, fix_seed=args.fix_seed)
                                            
    else:
        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme, fix_seed=args.fix_seed)
    
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()), tokenizer=tokenizer, device='cuda:0', 
                                            gamma=args.gamma, delta=args.delta, seeding_scheme=args.seeding_scheme,
                                            z_threshold=args.detection_z_threshold, normalizers=args.normalizers, fix_seed=args.fix_seed)

    file_path = f"output/{args.synonym_method}/gsm8k_completely_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_result_sampling{args.use_sampling}" if args.use_synonyms else f"output/gsm8k_327_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_result_sampling{args.use_sampling}"

    if args.remove_output_repetition: # # both watermark and x-mark
        file_path = f"output/{args.synonym_method}/gsm8k_327_completely_{args.model_name_or_path}_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_removeRepetition_result_sampling{args.use_sampling}" if args.use_synonyms else f"output/gsm8k_327_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_removeRepetition_result_sampling{args.use_sampling}"
    if args.filter_answer_line_when_detecting: # both watermark and x-mark
        file_path += f'filter_answer_line_when_detecting1'
    # file_path += '.jsonl'
    # file_path += '_shard0.jsonl'
    file_path += '_shard1.jsonl'
    # file_path += '_shard2.jsonl'
    # file_path += '_shard3.jsonl'
    # file_path += '_shard4.jsonl'
    # file_path += '_shard5.jsonl'
    # file_path += '_shard6.jsonl'
    # file_path += '_shard7.jsonl'
    
    f_output = open(file_path, 'w', encoding='utf-8')

    # generate and detect line by line. can we batch it?
    # for id in tqdm.tqdm(range(args.num_samples)):
    # 0
    # start, end = 0, 50
    start, end = 50, 100
    # start, end = 0, 100
    # start, end = 100, 200
    # start, end = 25, 50
    # start, end = 50, 75
    # start, end = 75, 100
    # start, end = 100, 125
    # start, end = 125, 150
    # start, end = 150, 175
    # start, end = 175, 200
    for id in tqdm.tqdm(range(start, end)):
        cur_res = {}
        # input_text = [next(dataset_itr)['text']]
        test = gsm8k_test[id]
        question = test['question'].strip()
        answer = test['answer'].strip()
        answer = re.sub(r'<<.*?>>', '', answer)
        final_answer = answer.split("#### ")[-1].strip()

        # input_text = 'Question: ' + question + "\nLet's think step by step:\n"
        input_text = 'Question: ' + question + "\nLet's think step by step:"
        # input_text = 'Question: ' + question + "\n"
        # input_text = 'Question: ' + question
        if args.few_shot:
            input_text = prompt + '\n\n' + input_text

        # generate with and without watermark
        decoded_output_without_watermark, decoded_output_with_watermark = generate(input_text, args, watermark_processor, model=model, tokenizer=tokenizer)

        decoded_output_without_watermark_detection, decoded_output_with_watermark_detection = copy.deepcopy(decoded_output_without_watermark), copy.deepcopy(decoded_output_with_watermark)

        wo_acc, w_acc = False, False
        if 'The answer is ' in decoded_output_without_watermark:
            without_watermark_final_answer = decoded_output_without_watermark.split('The answer is ')[-1].strip('.').strip()
            wo_acc = (without_watermark_final_answer == final_answer)
        else:
            final_line = decoded_output_without_watermark.split('\n')[-1]
            wo_acc = (final_answer in final_line)

        if 'The answer is ' in decoded_output_with_watermark:
            with_watermark_final_answer = decoded_output_with_watermark.split('The answer is ')[-1].strip('.').strip()
            w_acc = (with_watermark_final_answer == final_answer)
        else:
            final_line = decoded_output_with_watermark.split('\n')[-1]
            w_acc = (final_answer in final_line)

        if args.remove_output_repetition:
            decoded_output_without_watermark_detection = remove_repeated_substrings(decoded_output_without_watermark_detection)
            decoded_output_with_watermark_detection = remove_repeated_substrings(decoded_output_with_watermark_detection)

        if args.filter_answer_line_when_detecting:
            decoded_output_without_watermark_detection = filter_answer_line(decoded_output_without_watermark_detection)
            decoded_output_with_watermark_detection = filter_answer_line(decoded_output_with_watermark_detection)

        pt = PT(system_prompt="Please paraphrase the following text, altering the wording significantly yet preserving the original meaning and length, and promptly return only the paraphrased text without any additional content or affirmative responses.")
        # the first user message
        pt.add_user_message(decoded_output_with_watermark_detection)
        rewrite_prompt = pt.build_prompt()
        # print('rewrite_prompt = \n', rewrite_prompt)

        decoded_output_with_watermark_detection_paraphrased = paraphrase_response(rewrite_prompt, decoded_output_with_watermark_detection, args, paraphraser, tokenizer_paraphraser) 

        # print('source = \n', decoded_output_with_watermark_detection)      
        # print('paraphrased = \n', decoded_output_with_watermark_detection_paraphrased)  

        # detect
        # without_watermark_detection_result = detect(decoded_output_without_watermark, args, watermark_detector)
        # with_watermark_detection_result = detect(decoded_output_with_watermark, args, watermark_detector)
        without_watermark_detection_result = detect(decoded_output_without_watermark_detection, args, watermark_detector)
        with_watermark_detection_result = detect(decoded_output_with_watermark_detection, args, watermark_detector)
        with_watermark_detection_result_para = detect(decoded_output_with_watermark_detection_paraphrased, args, watermark_detector)

        # prefix = question
        prefix = input_text
        suffix = answer                                                                                            
        cur_res["prompt"] = prefix
        cur_res["suffix"] = suffix
        cur_res["wo_watermark_output"] = decoded_output_without_watermark
        cur_res["w_watermark_output"] = decoded_output_with_watermark
        cur_res["w_watermark_output_para"] = decoded_output_with_watermark_detection_paraphrased

        cur_res["wo_watermark_detection"] = without_watermark_detection_result[0]
        cur_res["w_watermark_detection"] = with_watermark_detection_result[0]
        cur_res["w_watermark_detection_para"] = with_watermark_detection_result_para[0]

        cur_res["wo_watermark_acc"] = wo_acc
        cur_res["w_watermark_acc"] = w_acc

        f_output.write(json.dumps(cur_res) + '\n')
        f_output.flush()

        if args.debug_mode:
            term_width=50
            print("#"*term_width)
            print(f"Sample {id} of {args.num_samples}")
            print("Question:")
            print(question)
            # print ('input_text:')
            # print (input_text)
            print('Answer:')
            print(answer)
            # print (f'Final answer: {final_answer}')

            print("#"*term_width)
            print("Output without watermark:")
            print(decoded_output_without_watermark)
            print (f'\nThe final answer is {cur_res["wo_watermark_acc"]}')
            # print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["wo_watermark_detection"])
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            print(decoded_output_with_watermark)
            print (f'\nThe final answer is {cur_res["w_watermark_acc"]}')
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["w_watermark_detection"])
            print("-"*term_width)

    return

if __name__ == "__main__":

    args = parse_args()

    main(args)
