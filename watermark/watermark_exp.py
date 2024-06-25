# coding=utf-8
import os
import argparse
from functools import partial
import numpy as np
import json
import torch
import tqdm
import math
import pprint
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          GPT2LMHeadModel
                          )
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
    
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from watermark_processor_syn import WatermarkLogitsProcessor_with_synonym, WatermarkDetector_with_synonym

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
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="gpt2_g0.5_d2.0_bl",
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
        "--use_synonyms",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--debug_mode",
        type=str2bool,
        default=False,
    )
    args = parser.parse_args()
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    return args

def load_model(args, model_name_or_path):
    """Load and return the model and tokenizer"""
    print(f"Loading model: {args.model_name_or_path}")
    if 'llama' not in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    args.is_seq2seq_model = any([(model_type in model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in model_name_or_path) for model_type in ["gpt","opt","bloom", "LLaMA", "llama", "llama2"]])

    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            if 'llama' in model_name_or_path:
                tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left') # left-padding for decoder-only model
                tokenizer.pad_token = -1
                tokenizer.bos_id = 1
                tokenizer.eos_id = 2
                if '7' or '13' in model_name_or_path:
                    # model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to('cuda:0')
                    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
                else:
                    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
                    # model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="balanced_low_0", torch_dtype=torch.float16)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    # print(f"Generating with {args}")
    if args.use_synonyms:
        watermark_processor = WatermarkLogitsProcessor_with_synonym(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme=args.seeding_scheme,
                                                        select_green_tokens=args.select_green_tokens)
    else:
        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

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

    if args.suppress_eos:
        gen_kwargs.update(dict(suppress_tokens=[tokenizer.eos_token_id]))

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

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_min_length).to(device)
    prefix = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    tokd_suffix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][:, args.prompt_min_length: args.prompt_min_length + args.max_new_tokens].to(device)
    suffix = tokenizer.batch_decode(tokd_suffix, skip_special_tokens=True)[0]

    # print (f"prefix length: {tokd_input['input_ids'].shape[-1]}")
    # print (f"suffix length: {tokd_suffix.shape[-1]}")


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
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (prefix,
            suffix,
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            ) 

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

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    if args.use_synonyms:
        watermark_detector = WatermarkDetector_with_synonym(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=args.gamma,
                                            seeding_scheme=args.seeding_scheme,
                                            device=device,
                                            tokenizer=tokenizer,
                                            z_threshold=args.detection_z_threshold,
                                            normalizers=args.normalizers,
                                            ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                            select_green_tokens=args.select_green_tokens)
    else:
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
                                        
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def load_exp_dataset(args):
    print ('Load realnewslike dataset...')
    dataset_name, dataset_config_name = 'c4', 'realnewslike'
    if not args.debug_mode:
        dataset = load_dataset(dataset_name, dataset_config_name, split="train", streaming=True)
    else:
        # dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
        dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
    ds_iterator = iter(dataset)
    return ds_iterator

def evaluate_generation_fluency(example: dict, 
                                oracle_model = None,
                                oracle_tokenizer = None):

    # pull out the required fields from the pipeline results
    # inputs_plus_baseline_output = f"{example['prompt']}{example['suffix']}"
    # baseline_output = f"{example['suffix']}"

    inputs_plus_wo_watermark_output = f"{example['prompt']}{example['wo_watermark_output']}"
    wo_watermark_output = f"{example['wo_watermark_output']}"

    inputs_plus_w_watermark_output = f"{example['prompt']}{example['w_watermark_output']}"
    w_watermark_output = f"{example['w_watermark_output']}"

    # add metrics
    # loss, ppl = compute_ppl_single(inputs_plus_baseline_output, baseline_output, oracle_model, oracle_tokenizer)
    # example["baseline_loss"] = loss
    # example["baseline_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_wo_watermark_output, wo_watermark_output, oracle_model, oracle_tokenizer)
    example["wo_watermark_loss"] = loss
    example["wo_watermark_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_w_watermark_output, w_watermark_output, oracle_model, oracle_tokenizer)
    example["w_watermark_loss"] = loss
    example["w_watermark_ppl"] = ppl

    # del any temp values
    return example

def compute_ppl_single(prefix_and_output_text = None, output_text = None, model = None, tokenizer = None):

    with torch.no_grad():
        tokd_inputs = tokenizer.encode(prefix_and_output_text, return_tensors="pt")
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenizer.encode(output_text, return_tensors="pt")
        tokd_inputs = tokd_inputs.to(model.device)

        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:, :tokd_labels.shape[1] - tokd_suffix.shape[1] + 1] = -100 # mask out the prefix

        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()

def main(args): 
    # load model and dataset
    dataset_itr = load_exp_dataset(args) # realnewslike
    model, tokenizer, device = load_model(args, args.model_name_or_path)

    # generate and detect line by line
    # can we batch it?
    res = []
    for id in tqdm.tqdm(range(args.num_samples)):
        cur_res = {}
        input_text = [next(dataset_itr)['text']]

        prefix, suffix, decoded_output_without_watermark, decoded_output_with_watermark = generate(input_text, 
                                                                                            args, 
                                                                                            model=model, 
                                                                                            device=device, 
                                                                                            tokenizer=tokenizer)
        cur_res["prompt"] = prefix
        cur_res["suffix"] = suffix
        cur_res["wo_watermark_output"] = decoded_output_without_watermark
        cur_res["w_watermark_output"] = decoded_output_with_watermark

        without_watermark_detection_result = detect(decoded_output_without_watermark, 
                                                    args, device=device, 
                                                    tokenizer=tokenizer)

        with_watermark_detection_result = detect(decoded_output_with_watermark, 
                                                    args, device=device, 
                                                    tokenizer=tokenizer)

        cur_res["wo_watermark_detection"] = without_watermark_detection_result[0]
        cur_res["w_watermark_detection"] = with_watermark_detection_result[0]
        res.append(cur_res)

        if args.debug_mode:
            term_width=50
            print("#"*term_width)
            print(f"Sample {id} of {args.num_samples}")
            print("Prompt:")
            print(input_text)

            print("#"*term_width)
            print("Output without watermark:")
            # print(decoded_output_without_watermark)
            # print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["wo_watermark_detection"])
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            # print(decoded_output_with_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            print(cur_res["w_watermark_detection"])
            print("-"*term_width)

    # Load the oracle model for PPL measurement
    # Assume on single GPU and need to free orig model memory for oracle model
    if model is not None and args.oracle_model_name != args.model_name_or_path:
        model = model.to(torch.device("cpu"))
        del model
        oracle_model, oracle_tokenizer, device = load_model(args, args.oracle_model_name)
    else:
        oracle_model, oracle_tokenizer = model, tokenizer

    oracle_model.eval()

    final_res = []
    avg_wo_watermark_ppl, avg_w_watermark_ppl = 0, 0
    avg_z_wo_watermark, avg_z_w_watermark = 0, 0
    for example in tqdm.tqdm(res):
        res_example = evaluate_generation_fluency(example, oracle_model, oracle_tokenizer)
        # pprint(res_example)
        final_res.append(res_example)
        
    avg_wo_watermark_ppl = np.mean([exp['wo_watermark_ppl'] for exp in final_res])
    avg_w_watermark_ppl = np.mean([exp['w_watermark_ppl'] for exp in final_res])
    avg_z_wo_watermark = np.mean([float(exp['wo_watermark_detection'][3][-1]) for exp in final_res])
    avg_z_w_watermark = np.mean([float(exp['w_watermark_detection'][3][-1]) for exp in final_res])
    print(f"Average w watermark z: {avg_z_w_watermark}")
    print(f"Average w watermark ppl: {avg_w_watermark_ppl}\n")

    print(f"Average wo watermark z: {avg_z_wo_watermark}")
    print(f"Average wo watermark ppl: {avg_wo_watermark_ppl}")

    final_res.append({"avg_wo_watermark_ppl": avg_wo_watermark_ppl,
                      "avg_w_watermark_ppl": avg_w_watermark_ppl,
                      "avg_z_wo_watermark": avg_z_wo_watermark,
                      "avg_z_w_watermark": avg_z_w_watermark})
    
    with open(f"results/{args.exp_name}_results.json", "w") as f:
        json.dump(final_res, f, indent=4)

   
    return

if __name__ == "__main__":

    args = parse_args()

    main(args)
