# coding=utf-8
import os
import csv
import argparse
import sys, argparse, time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs':np.get_include()})
from levenshtein import levenshtein
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
        "--data_shard_id",
        type=int,
        default=-1
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

def generate_shift(model, tokenizer, prompt, vocab_size, n, m, key):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand() for _ in range(n*vocab_size)]).view(n,vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs,xi[(shift+i)%n,:]).to(model.device)
        if token == tokenizer.eos_token_id:
            break
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def exp_sampling(probs,u):
    return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

def generate(prompt, args, watermark_processor, tokenizer, model):
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
    # output_with_watermark = generate_with_watermark(**tokd_input)
    output_with_watermark = generate_shift(model, tokenizer, tokd_input['input_ids'], len(tokenizer), args.max_new_tokens, args.max_new_tokens, args.generation_seed)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    # print (f"len of wo watermark: {len(output_without_watermark[0])}")
    # print (f"len of w watermark: {len(output_with_watermark[0])}")

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0].split('\n')[0] # generate multiple answers, but only take the first one
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0].split('\n')[0]

    return (
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            ) 

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect_stanford(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect_stanford(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)

def detect_stanford(tokens,n,k,xi,gamma=0.0):
    '''
    k: block size 
    n: key length
    '''
    m = len(tokens)
    n = len(xi)

    k = m # default: m = n = k

    # A = np.empty((m-(k-1),n))
    # A = np.empty((m))
    # for i in range(m):
        # for j in range(n):
            # A[i][j] = levenshtein(tokens[i:i+k], xi[(j+np.arange(k))%n],gamma)
        # A[i] = levenshtein(tokens, xi, gamma)

    # return np.min(A)

    min_len = min(len(tokens), len(xi))
    return levenshtein(tokens[:min_len], xi[:min_len], gamma) # 0.27/0.0099=27.3
    # return sum([xi[i][token] for i, token in enumerate(tokens)]) # 1.0/0.079=12.7

def detect(input_text, tokenizer, args):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    if len(input_text)-1 > 2:
        # score_dict = watermark_detector.detect(input_text)
        tokens = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        pval = permutation_test(tokens, args.generation_seed, args.max_new_tokens, len(tokens),len(tokenizer))
        score_dict = {'z_score': pval, 'p_value': pval, 'confidence': 1.0}
        output = list_format_scores(score_dict, 4.0)
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

def get_shard_range(num_samples, num_shards, shard_id):
    samples_per_shard = num_samples // num_shards
    start = shard_id * samples_per_shard
    # 如果是最后一个shard，确保包含所有剩余的样本
    if shard_id == num_shards - 1:
        end = num_samples
    else:
        end = start + samples_per_shard
    return start, end

def main(args): 
    # load model and dataset
    # dataset_itr = load_exp_dataset(args) # realnewslike
    truthfulqa_test = read_truthfulqa('data/TruthfulQA.csv')
    args.num_samples = min(len(truthfulqa_test), args.num_samples)

    watermark_processor, watermark_detector = None, None
    
    file_path = f"output/{args.synonym_method}/stanford_new_truthfulqa_completely_{args.model_name_or_path}_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_result.jsonl" if args.use_synonyms else f"output/stanford_new_truthfulqa_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_result.jsonl"
    if args.remove_output_repetition:
        file_path = f"output/{args.synonym_method}/stanford_new_truthfulqa_completely_{args.model_name_or_path}_sg{args.synonym_gamma}_g{args.gamma}_d{args.delta}_removeRepetition_result.jsonl" if args.use_synonyms else f"output/stanford_new_truthfulqa_{args.model_name_or_path}_g{args.gamma}_d{args.delta}_removeRepetition_result.jsonl"
    
    calculated_results = []
    if os.path.isfile(file_path):
        if os.path.getsize(file_path) > 0:
            with open(file_path, "r") as f:
                for line in f:
                    calculated_results.append(json.loads(line.strip()))

    model, tokenizer = None, None
    if len(calculated_results) == 1319:
        tokenizer = load_tokenizer(args.model_name_or_path)
    else:
        model, tokenizer = load_model(args.model_name_or_path)

    state = 'a'
    # if len(calculated_results) == 1319 and args.data_shard_id != -1:
    if args.data_shard_id != -1:
        file_path += f'_shard{args.data_shard_id}'
        state = 'w'

    f_output = open(file_path, state, encoding='utf-8')

    start, end = 0, args.num_samples
    if args.data_shard_id != -1:
        start, end = get_shard_range(args.num_samples, 8, args.data_shard_id)

    for id in tqdm.tqdm(range(start, end)):
        cur_res = {}
        # input_text = [next(dataset_itr)['text']]
        test = truthfulqa_test[id]
        assert len(test) == 7, len(test)
        topic = test[1]
        question = test[2].strip()
        best_answer, correct_answers, incorrect_answers = test[3], test[4], test[5]

        input_text = f"Generate an answer for the given question.\nQuestion: {question}\nAnswer:"
        if args.few_shot:
            input_text = f"{demonstrations['qa'].strip()}\n\nQ: {question}\nA:"

        # generate with and without watermark
        decoded_output_without_watermark, decoded_output_with_watermark = generate(input_text, args, watermark_processor, model=model, tokenizer=tokenizer)

        if args.remove_output_repetition:
            decoded_output_without_watermark = remove_repeated_substrings(decoded_output_without_watermark)
            decoded_output_with_watermark = remove_repeated_substrings(decoded_output_with_watermark)

        # prefix = question
        prefix = input_text
        suffix = best_answer                                                                                            
        cur_res["prompt"] = prefix
        cur_res["suffix"] = suffix
        cur_res["wo_watermark_output"] = decoded_output_without_watermark
        cur_res["w_watermark_output"] = decoded_output_with_watermark

        if len(decoded_output_without_watermark.split()) > 1 and len(decoded_output_with_watermark.split()) > 1:
            # detect
            without_watermark_detection_result = detect(decoded_output_without_watermark, tokenizer, args)
            with_watermark_detection_result = detect(decoded_output_with_watermark, tokenizer, args)

            cur_res["wo_watermark_detection"] = without_watermark_detection_result[0]
            cur_res["w_watermark_detection"] = with_watermark_detection_result[0]

        f_output.write(json.dumps(cur_res) + '\n')
        f_output.flush()

        if args.debug_mode:
            term_width=50
            print("#"*term_width)
            print(f"Sample {id} of {args.num_samples}")
            print("Question:")
            print(question)
            print('Answer:')
            print(best_answer + correct_answers)

            print("#"*term_width)
            print("Output without watermark:")
            print(decoded_output_without_watermark)
            # print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["wo_watermark_detection"])
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            print(decoded_output_with_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["w_watermark_detection"])
            print("-"*term_width)

    return

if __name__ == "__main__":

    args = parse_args()

    main(args)
