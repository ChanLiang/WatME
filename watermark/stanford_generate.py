import os, argparse

import os, sys, argparse, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng

import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs':np.get_include()})
from levenshtein import levenshtein

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)


def detect(tokens,n,k,xi,gamma=0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],gamma)

    return np.min(A)

def generate_shift(model,prompt,vocab_size,n,m,key):
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
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def exp_sampling(probs,u):
    return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print ('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)

    tokens = tokenizer.encode(args.prompt, return_tensors='pt', truncation=True, max_length=2048)

    watermarked_tokens = generate_shift(model,tokens,len(tokenizer),args.n,args.m,args.key)[0]
    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

    print(watermarked_text)

    t0 = time.time()
    tokens = tokens.numpy()[0]
    pval = permutation_test(tokens, args.key, args.n, len(tokens),len(tokenizer))
    print('p-value: ', pval)
    print(f'(elapsed time: {time.time()-t0}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--model',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--prompt',default='',type=str,
            help='an optional prompt for generation')
    parser.add_argument('--m',default=80,type=int,
            help='the requested length of the generated text')
    parser.add_argument('--n',default=256,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='a key for generating the random watermark sequence')
    parser.add_argument('--seed',default=0,type=int,
            help='a seed for reproducibile randomness')

    main(parser.parse_args())
