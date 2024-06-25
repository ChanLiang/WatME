import torch
import pickle
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from wordfreq import zipf_frequency
import nltk
# nltk.download('words')
from nltk.corpus import words

# # # 加载tokenizer和模型
tokenizer = LlamaTokenizer.from_pretrained('/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/')
# model = LlamaForCausalLM.from_pretrained("/apdcephfs/share_1594716/chenliang/cache/vicuna-7b-v1.5-16k/", device_map="auto", torch_dtype=torch.float16)

# # 获取嵌入矩阵，并转移到CPU上
# embeddings = model.get_input_embeddings()

# # Convert embeddings to numpy and detach from computation graph
# embeddings = embeddings.weight.detach().cpu().numpy()

# # 获取词汇表
# vocab = tokenizer.get_vocab()

# 创建一个新的字典，存储每个token和其对应的embedding
def is_valid_word(word): 
    '''32000 --> 3873'''
    if word[0] == '_':
        word = word[1:]
    word = word.lower()
    res = word.isalpha() and word in words.words()
    # if res:
    #     print (word)
    return res

def is_common_word(word, language='en', min_frequency=3.5):
    # Zipf 值 3 的单词每百万单词出现一次 --》 4976
    # 3.5 --> 4011
    # Zipf 值 4 的单词每十万单词出现一次 --》 2999
    if word[0] == '_':
        word = word[1:]
    word = word.lower()
    freq = zipf_frequency(word, language)
    # return (freq > min_frequency, freq)
    return freq > min_frequency

# # embedding_dict = {id: embeddings[id] for token, id in tqdm(vocab.items()) if is_valid_word(token)}
# embedding_dict = {id: embeddings[id] for token, id in tqdm(vocab.items()) if is_common_word(token)}
# print(len(embedding_dict)) 

# # 存储embedding字典
# with open('vicuna_7b_embedding_dict_filter.pickle', 'wb') as handle:
#     pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 从文件中加载字典
with open('vicuna_7b_embedding_dict_filter.pickle', 'rb') as handle:
    embedding_dict = pickle.load(handle)

# 获取所有的embedding和对应的id
ids = list(embedding_dict.keys())
embeddings = np.array([embedding_dict[id] for id in ids])

# 创建一个faiss索引
dimension = embeddings.shape[-1]  # 获取embedding的维度
index = faiss.IndexFlatIP(dimension)  # 使用内积索引

# 使用GPU资源
res = faiss.StandardGpuResources()  # Default GPU resource object.
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 在GPU上创建索引

# 对嵌入进行归一化
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
embeddings_norm = embeddings_norm.astype('float32')

gpu_index.add(embeddings_norm)  # 添加数据到索引

# 对每个id，找出最相近的k个id
k = 10 
alpha = 0.8
distances, nearest_indices = gpu_index.search(embeddings_norm, k)

# 创建一个字典，key是id，value是最相近的id列表
nearest_ids_dict = {}
for idx, nearest, distance in zip(ids, nearest_indices, distances):
    # print (idx, nearest, distance)
    # 只保留那些距离小于alpha的id，并且排除id自身
    # nearest_ids = [ids[i] for i, d in zip(nearest, distance) if d > alpha and ids[i] != idx]
    nearest_ids = [ids[i] for i, d in zip(nearest, distance) if d > 0.5 and ids[i] != idx]
    nearest_ids_dict[idx] = nearest_ids

# 保存字典到文件
with open('vicuna_7b_nearest_ids_dict.pickle', 'wb') as handle:
    pickle.dump(nearest_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 从文件中加载字典
with open('vicuna_7b_nearest_ids_dict.pickle', 'rb') as handle:
    nearest_ids_dict = pickle.load(handle)

def save_nearest_tokens(nearest_ids_dict, tokenizer, filename):
    nearest_tokens_dict = {}

    for id, nearest_ids in nearest_ids_dict.items():
        token = tokenizer.convert_ids_to_tokens(id)
        nearest_tokens = tokenizer.convert_ids_to_tokens(nearest_ids)
        nearest_tokens_dict[token] = nearest_tokens

    with open(filename, 'w') as f:
        for token, nearest_tokens in nearest_tokens_dict.items():
            f.write(f"{token}: {', '.join(nearest_tokens)}\n")

save_nearest_tokens(nearest_ids_dict, tokenizer, 'vicuna_7b_nearest_tokens.txt')