import json
import tqdm
import os
import pickle
import nltk
import torch
import string
import argparse
import unicodedata
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from get_synonmy_by_youdao import get_synonyms

nltk.data.path.append('/apdcephfs/share_1594716/chenliang/cache/nltk')

word_set = set(words.words())
wnl = WordNetLemmatizer()

def get_lemma(word):
    word = word.strip()
    lemma = wnl.lemmatize(word.lower())  # 默认还原为动词原型
    return lemma

def is_valid_english_word(word):
    # 不会考虑词形变化，比如复数形式或者动词的过去式，这些可能需要额外的处理
    word = get_lemma(word.lstrip("▁").lower()).strip()
    if len(word) > 1 and word and (wordnet.synsets(word) or word in word_set):  # 如果在词典中，但是可能是大写字母开头，再检查一遍
        return True
    else:
        return False

def get_vocab_dict(model_name, vocab_path):
    assert os.path.splitext(vocab_path)[1] == '.json', "File is not a JSON file"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    if model_name in ['llama', 'llama2', 'vicuna']:
        return vocab['model']['vocab']
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_name):
    if 'vicuna' in model_name:
        print ("loading vicuna-13b-v1.5-16k")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-13b-v1.5-16k/", padding_side='left')
        tokenizer.pad_token = -1
        tokenizer.bos_id = 1
        tokenizer.eos_id = 2
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-13b-v1.5-16k/", device_map="auto", torch_dtype=torch.float16)
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
    elif 'llama1-65b' in model_name:
        print ("loading llama")
        tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", padding_side='left')
        model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/65B/", device_map="auto", torch_dtype=torch.float16)
    else:
        return ValueError('error!')
    return tokenizer, model

def parse_demonstrations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def infer_valid_words(model, tokenizer, id_token_list, bz, infer_gpu, debug=False):
    print ('get_valid_words...')
    demon_dic = parse_demonstrations(f"../demonstrations/valid_word.json")
    cur_batch, cur, valid_id_token_list = [], 0, []
    for i in tqdm.tqdm(range(len(id_token_list))):
        idx, word = id_token_list[i]
        word = word.lstrip("▁").strip() 

        prompt = demon_dic['task_description'] + '\n\n' 
        for j in range(1, len(demon_dic)):
            prompt += demon_dic[f'demonstration{j}'].strip() + '\n\n' 

        # prompt += f"Input word: {word}\n"
        # prompt += f"Is this a valid English word?"
        prompt += f"Can '{word}' be considered a valid English word?"
        cur_batch.append(prompt)

        if len(cur_batch) == bz or (i == len(id_token_list) - 1 and len(cur_batch)):
            inputs = tokenizer(cur_batch, return_tensors="pt", padding=True)
            inputs = inputs.to(infer_gpu)

            output = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=1,
            )
            if type(model) == LlamaForCausalLM: # for decoder-only model
                output = output[:,inputs["input_ids"].shape[-1]:]

            responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            assert len(responses) == len(cur_batch)

            for j, response in enumerate(responses):
                # print (response)
                response = response.split('Can ')[0].strip() # generate multiple examples
                validity = response.strip()

                if debug:
                    print ('=' * 50)
                    # print (cur_batch[j])
                    # print ('-' * 50)
                    print (cur_batch[j].split('Can ')[-1].strip())
                    print ('-' * 50)
                    print (response)
             
                if validity.lower() == 'yes':
                    valid_id_token_list.append(id_token_list[cur])
                cur += 1
            cur_batch = []

    assert cur == len(id_token_list), (cur, len(id_token_list))

    return valid_id_token_list

def infer_synonyms_by_lm(model, tokenizer, id_token_list, bz, infer_gpu, debug=False):
    print ('infer_synonyms_by_lm...')
    fw = open('output/lm_synonyms.txt', 'a')
    demon_dic = parse_demonstrations(f"../demonstrations/synonyms.json")
    cur_batch, cur = [], 0
    for i in tqdm.tqdm(range(len(id_token_list))):
        idx, word = id_token_list[i]
        word = word.lstrip("▁").strip() 

        prompt = demon_dic['task_description'] + '\n\n' 
        for j in range(1, len(demon_dic)):
            prompt += "Input: " + demon_dic[f'demonstration{j}'].strip() + '\n\n' 

        prompt += f"Input: {word}\n"
        prompt += f"Synonyms:"
        cur_batch.append(prompt)

        if len(cur_batch) == bz or (i == len(id_token_list) - 1 and len(cur_batch)):
            inputs = tokenizer(cur_batch, return_tensors="pt", padding=True)
            inputs = inputs.to(infer_gpu)

            output = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=10,
            )
            if type(model) == LlamaForCausalLM: # for decoder-only model
                output = output[:,inputs["input_ids"].shape[-1]:]

            responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            assert len(responses) == len(cur_batch)

            for j, response in enumerate(responses):
                # print (response)
                response = response.split('Input word:')[0].strip() # generate multiple examples
                # synonyms = list(set(response.strip().split('\n'))) # generate multiple examples
                synonyms = list(set(response.strip().split('\t'))) # generate multiple examples
                if len(synonyms) == 1:
                    synonyms = synonyms[0].split()
                    synonyms = list(set(synonyms))
                # synonyms = [synonym.strip() for synonym in synonyms if synonym.strip() != word and is_valid_english_word(synonym.strip())]
                cur_word1 = id_token_list[cur][1].lstrip("▁").strip() 
                cur_word = id_token_list[cur][1] 
                synonyms = [synonym.strip() for synonym in synonyms if synonym.strip() != cur_word1]
                if cur_word[0] == "▁":
                    synonyms = [f"▁{synonym.strip()}" for synonym in synonyms]
               
                if debug:
                    print ('=' * 50)
                    # print (cur_batch[j])
                    # print ('-' * 50)
                    # print (cur_batch[j].split('Input word:')[-1].strip())
                    print (cur_word)
                    # print ('-' * 50)
                    print ('response = ', response)
                    print ('synonyms = ', synonyms)
             

                id_token_list[cur].append(synonyms)
                cur += 1
                fw.write(f"{cur_word}\t{synonyms}\n")
            cur_batch = []

    assert cur == len(id_token_list), (cur, len(id_token_list))

    return id_token_list

def infer_synonyms_by_chat(model, tokenizer, id_token_list, bz, infer_gpu, debug=False):
    print ('get_abs_synonyms...')
    fw = open('output/chat_synonyms.txt', 'a')
    demon_dic = parse_demonstrations(f"../demonstrations/synonyms.json")
    cur_batch, cur = [], 0
    for i in tqdm.tqdm(range(len(id_token_list))):
        idx, word = id_token_list[i]
        word = word.lstrip("▁").strip() 

        # system prompt
        prompt = "<s>[INST] <<SYS>>\n" + demon_dic['task_description'].strip() + "\n" + "<</SYS>>" + "\n\n"
        for j in range(1, len(demon_dic)):
            input_word, output= demon_dic[f'demonstration{j}'].split("\nSynonyms:") 
            output = output.replace('\t', ', ')
            # input_word = input_word.lstrip("Input word: ").strip()
            prompt += f"Give me distinct synonyms for the word '{input_word.strip()}'. [/INST] Sure, here are some synonyms for '{input_word.strip()}': {output.strip()}. </s><s>[INST] "
            # prompt += f"'Word: {input_word.strip()}' [/INST] Synonyms: {output.strip()} </s><s>[INST] "

        # prompt += f"{word} [/INST]" 
        prompt += f"Give me distinct synonyms for the word '{word.strip()}'. [/INST]"
        # prompt += f"'Word: {word.strip()}' [/INST]"
        cur_batch.append(prompt)

        if len(cur_batch) == bz or (i == len(id_token_list) - 1 and len(cur_batch)):
            inputs = tokenizer(cur_batch, return_tensors="pt", padding=True)
            # inputs = tokenizer.encode(cur_batch, return_tensors="pt", padding=True)
            inputs = inputs.to(infer_gpu)

            output = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                # inputs,
                max_new_tokens=30,
                # max_new_tokens=35,
            )

            if type(model) == LlamaForCausalLM: # for decoder-only model
                output = output[:,inputs["input_ids"].shape[-1]:]

            responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            # print (responses)
            assert len(responses) == len(cur_batch)

            for j, response in enumerate(responses):
                s_response = response
                cur_word1 = id_token_list[cur][1].lstrip("▁").strip() 
                cur_word = id_token_list[cur][1] 
                # print (cur_word1, response)

                # response = response.split("Word: ")[0].strip() # generate multiple examples
                # response = response.lstrip("Synonyms: ").strip()

                response = response.split("Give me distinct synonyms for the word")[0].strip()  # generate multiple examples

                response = response.split(":")[1].strip() if len(response.split(":")) > 1 else response.split(":")[0].strip()
                # response = response.split("Generate distinct synonyms for the input word.")[0].strip() # generate multiple examples

                synonyms = []
                # synonyms = list(set(response.strip().split('\t'))) 
                if ',' in response:
                    synonyms = list(set(response.strip().split(',')))
                else:
                    synonyms = list(set(response.strip().split('\n')))
                synonyms = [synonym.strip() for synonym in synonyms if synonym.strip() != cur_word1 and synonym.strip()]
                if cur_word[0] == "▁":
                    # synonyms = [synonym.split('.')[1].strip() for synonym in synonyms]
                    synonyms = [f"▁{synonym.strip()}" for synonym in synonyms]
               
                if debug:
                    print ('=' * 50)
                    print ('input = ', cur_batch[j])
                    print ('input_word = ', cur_word)
                    # print ('-' * 50)
                    print ('s_response = ', s_response)
                    print ('response = ', response)
                    print ('synonyms = ', synonyms)
             

                id_token_list[cur].append(synonyms)
                cur += 1
                fw.write(f"{cur_word}\t{synonyms}\n")
            cur_batch = []

    assert cur == len(id_token_list), (cur, len(id_token_list))

    return id_token_list

def infer_synonyms_by_youdao(id_token_list, debug=False):
    fw = open('output/youdao_synonyms.txt', 'a')
    for i in tqdm.tqdm(range(len(id_token_list))):
        idx, word = id_token_list[i]
        word1 = word.lstrip("▁").strip() 
        synonyms = get_synonyms(word=word1)
        if word.isupper():
            synonyms = [synonym.upper() for synonym in synonyms]
        else:
            synonyms = [synonym[0].upper() + synonym[1:] if word[0].isupper() else synonym[0].lower() + synonym[1:] for synonym in synonyms]
        if word[0] == "▁":
            synonyms = [f"▁{synonym.strip()}" for synonym in synonyms]

        if debug:
            print (word, synonyms)
        id_token_list[i].append(synonyms)
        fw.write(f"{word}\t{synonyms}\n")
    fw.close()
    return id_token_list

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='llama2_supervised_synonym_mining')
    parser.add_argument("--debug", type=boolean_string, default=True)

    parser.add_argument('--watermark_model', type=str, default="llama2")
    parser.add_argument('--vocab_path', type=str, default="/apdcephfs/share_1594716/chenliang/cache/Llama-2-13b-hf/tokenizer.json")

    parser.add_argument('--synonym_model', type=str, default="vicuna")
    parser.add_argument('--method', type=str, default="lm")

    parser.add_argument('--valid_bz', type=int, default=96, help='batch size')
    # parser.add_argument('--valid_bz', type=int, default=128, help='batch size')
    parser.add_argument('--bz', type=int, default=32, help='batch size')
    parser.add_argument('--infer_gpu', type=int, default=0)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    tokenizer, model = None, None

    # load vocab of model to be watermarked
    token_id_dict = get_vocab_dict(args.watermark_model, args.vocab_path)
    id_token_dict = {v: k for k, v in token_id_dict.items()}
    id_token_list = list(id_token_dict.items())
    print (f"Number of tokens: {len(token_id_dict)}")

    # saved files for valid words and synonyms
    output_dir = f'./output/{args.exp_name}'
    os.makedirs(output_dir, exist_ok=True)  

    valid_words_file = os.path.join(output_dir, 'valid_id_token_list_2c.pkl')
    # synonyms_file = os.path.join(output_dir, 'id_token_synonyms_list_2c.pkl')
    synonyms_file = os.path.join(output_dir, 'id_token_synonyms_list_2c.txt')

    ''' 1. load or infer valid/complete words in the vocab'''
    valid_id_token_list, id_token_synonyms_list = [], []
    if os.path.exists(valid_words_file):
        print("Loading previously inferred valid words...")
        with open(valid_words_file, 'rb') as f:
            valid_id_token_list = pickle.load(f)
        print (f"Number of valid tokens: {len(valid_id_token_list)}") # 11875
    else:
        ''' filter twice to get valid words in vocab '''
         # 1. filter subwords by dictionary
        can_valid_token_id_dict, can_valid_id_token_dict = {}, {}
        for token, index in tqdm.tqdm(token_id_dict.items()):
            if is_valid_english_word(token):
                can_valid_token_id_dict[token] = index
                can_valid_id_token_dict[index] = token
        can_id_token_list = list(can_valid_id_token_dict.items())
        print (f"Number of valid tokens after filtering by diction: {len(can_id_token_list)}") # 1c: 14627; 2c: 14409
        # 2. filter subwords by llm
         # load LLM for synonyms inference
        tokenizer, model = load_model(args.synonym_model)
        valid_id_token_list = infer_valid_words(model, tokenizer, can_id_token_list, args.valid_bz, args.infer_gpu, args.debug)
        print(f"Number of valid words after filtering by llm: {len(valid_id_token_list)}") # 1c 14009, 2c 11875
        # save the inferred valid words
        with open(valid_words_file, 'wb') as f:
            pickle.dump(valid_id_token_list, f)

    ''' 2. load or infer synonyms of valid words '''
    if os.path.exists(synonyms_file):
        print("Loading previously inferred synonyms...")
        if synonyms_file[-3:] == 'pkl':
            with open(synonyms_file, 'rb') as f:
                id_token_synonyms_list = pickle.load(f)
        else:
            print ('load .txt file...')
            with open(synonyms_file, 'r') as f:
                for i, line in enumerate(f):
                    # try:
                    token, synonyms_list = line.strip().split('\t')
                    # except:
                        # print (i, line)
                    synonyms_list = eval(synonyms_list)
                    token = token.strip()
                    id_token_synonyms_list.append([token_id_dict[token], token, synonyms_list]) # [262, 'in', ['within', 'inside', 'modern', 'fashionable', 'tony']]
            # print (len(id_token_synonyms_list))
            # print (id_token_synonyms_list[0])
            # print (id_token_synonyms_list[-1])

    else:
        valid_id_token_list = [[idx, token] for idx, token in valid_id_token_list]
        # infer synonyms
        id_token_synonyms_list = []
        if args.method == 'lm':
            id_token_synonyms_list = infer_synonyms_by_lm(model, tokenizer, valid_id_token_list, args.bz, args.infer_gpu, args.debug)
        elif args.method == 'youdao':
            id_token_synonyms_list = infer_synonyms_by_youdao(valid_id_token_list, args.debug)
        else:
            id_token_synonyms_list = infer_synonyms_by_chat(model, tokenizer, valid_id_token_list, args.bz, args.infer_gpu, args.debug)
        # save the inferred synonyms
        with open(synonyms_file, 'wb') as f:
            pickle.dump(id_token_synonyms_list, f)
    
    # map back to vocab 
    assert len(id_token_synonyms_list[0]) == 3, (len(id_token_synonyms_list[0]))
    for i in range(len(id_token_synonyms_list)):
        final = []
        for token in id_token_synonyms_list[i][-1]:
            if token in token_id_dict:
                final.append(token)
        id_token_synonyms_list[i].append(final)
    id_token_synonyms_final_list = id_token_synonyms_list

    # save for human evaluation
    # print (len(id_token_synonyms_final_list), id_token_synonyms_final_list[0]) # 1, [262, 'in', ['toward', 'within', 'into', 'inside'], ['into', 'inside']]
    with open(f"./{output_dir}/humanview", "w") as f:
        for i in range(len(id_token_synonyms_final_list)):
            # print (i)
            idx, token, synonyms, final_synonyms = id_token_synonyms_final_list[i][0:4]
            f.write(f"{idx}\t{token}   ||   {synonyms}   ||   {final_synonyms}\n")

    # form a 2D idx list [cluster_id, word_id], and filter out those lacking synonyms
    # consider merge two clusters if they share some words
    clusters, clusters_filtered = [], [] # 1. 不管重复；2. 合并和去重
    st = set()
    for i in range(len(id_token_synonyms_final_list)):
        token, final_synonyms = id_token_synonyms_final_list[i][1], id_token_synonyms_final_list[i][-1]
        # print (token, final_synonyms)
        cluster = list(set([token_id_dict[token]] + [token_id_dict[s] for s in final_synonyms]))
        if len(cluster) > 1: 
            clusters.append(cluster) # [[id1, id2...], [id3, id4...]]
        
        # filtered_cluster = [id for id in cluster if id not in st]
        # 尝试合并到和之前的重合度词最多、且重合个数大于1的cluster中去，否则过滤掉和之前cluster重复的词
        max_id, max_num = -1, 0
        for j in range(len(clusters_filtered)):
            overlap = len(set(cluster) & set(clusters_filtered[j]))
            #if overlap > 0:
                #print (f"overlap: {overlap} || {[id_token_dict[idx] for idx in cluster]} || {[id_token_dict[idx] for idx in clusters_filtered[j]]}")
            if overlap > max_num:
                max_id, max_num = j, overlap
        
        if max_num > 1:
            # merge two set
            clusters_filtered[max_id].update(cluster)
            #st.update(cluster)
        else:                
            filtered_cluster = set([id for id in cluster if id not in st])
            if len(filtered_cluster) > 1:
                clusters_filtered.append(filtered_cluster)
                #st.update(filtered_cluster)
        st.update(cluster)
        

    clusters_size = sum(len(sublist) for sublist in clusters)
    clusters_filtered_size = sum(len(sublist) for sublist in clusters_filtered)
    clusters_filtered = [list(sublist) for sublist in clusters_filtered]
    # print (type(clusters_filtered[0]))
    print (f"Number of clusters: {len(clusters)}, size: {clusters_size}")
    print (f"Number of filtered clusters: {len(clusters_filtered)}, size: {clusters_filtered_size}")

    # save for watermarking: a 2D idx list [cluster_id, word_id]
    with open(f"./{output_dir}/2d_list.json", "w") as f:
        json.dump(clusters, f)

    with open(f"./{output_dir}/completely_filtered_2d_list1.json", "w") as f:
        json.dump(clusters_filtered, f)
        

if __name__ == "__main__":
    main()