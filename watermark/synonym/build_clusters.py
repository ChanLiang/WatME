import json
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import tqdm

# 创建WordNetLemmatizer对象
wnl = WordNetLemmatizer()

# 定义一个函数，输入一个单词并返回其原型
def get_lemma(word):
    word = word.strip('\u0120')
    lemma = wnl.lemmatize(word.lower())  # 默认还原为动词原型
    return lemma

def get_all_related_words(w, vocab):
    related_words_ids = []
    for w2, idx in vocab.items(): # |V|
        if w2 == w:
            related_words_ids.append((w2, idx))
            continue
        
        w = w.lower()
        w2_ = get_lemma(w2)
        # if w2_ in w or w in w2_:
        if w2_ == w or w2_ == get_lemma(w):
            related_words_ids.append((w2, idx))

    return related_words_ids


def find_seed_in_vocab(w, vocab):
    if w in vocab:
        return (w, vocab[w])

    w = w.lower()
    for w2, idx in vocab.items(): # |V|
        w2_ = w2.strip('\u0120').lower()
        w2_lemma = get_lemma(w2_)
        # if w2_ in w or w in w2_ or w2_lemma in w or w in w2_lemma:
        if w2_ == w or w2_lemma == w:
        # if w2_ == w:
            return (w2, idx)
    
    # raise Exception(f'{w} not found')
    return []
        

vocab = {} # {word: index}
with open("vocab.json", "r") as f:
    vocab = json.load(f)

syn_dict = {}
st1 = set()
syn_file = 'res_syn_3c_only_yd'
with open(syn_file, 'r', encoding='utf-8') as r:
    for line in r:
        parts = line.strip().split('\t')
        assert len(parts) == 4, line
        # syn_list = eval(parts[-1])
        syn_list = eval(parts[-2]) # 包含了可能不在vocab中的同义词
        word = parts[1].strip()
        if not syn_list:
            continue
        syn_dict[word] = syn_list
        st1.add(word)
        st1 |= set(syn_list)

print (len(syn_dict), len(st1))

new_syn_dict = {}
st2 = set()
for k, v in syn_dict.items():
    li = []
    for syn in v:
        # if syn not in syn_dict or k in syn_dict[syn] or get_lemma(k) in syn_dict[k] or get_lemma(k) in [get_lemma(e) for e in syn_dict[k]]:
        if syn in syn_dict and (k in syn_dict[syn] or get_lemma(k) in syn_dict[syn] or get_lemma(k) in [get_lemma(e) for e in syn_dict[syn]]):
            li.append(syn)
    if li:
        new_syn_dict[k] = list(set(li))
        # print (k, v, new_syn_dict[k])
    # if len(li) < len(v):
        # print (k, v, li)
        st2.add(k)
        st2 |= set(li)

print (len(new_syn_dict), len(st2))
'''
14884 15501
7350 7983
'''

# new_syn_dict = syn_dict
fw = open('synonym_clusters.txt', 'w', encoding='utf-8')
fw1 = open('synonym_clusters.json', 'w', encoding='utf-8')

clusters = []
for word, syn_list in tqdm.tqdm(new_syn_dict.items()):
    # print (word, syn_list)
    try:
        seed_words = list(set(syn_list + [word]))
        # enlarge seed_words
        # cluster = [find_seed_in_vocab(w, vocab) for w in seed_words] # all in vocab
        cluster = []

        for w in seed_words:
            related_words_ids = get_all_related_words(w, vocab) # 也包括w自己
            cluster.append(related_words_ids)

        culster = [pair for pair in cluster if pair]
        fw.write(str(cluster).strip() + '\n')
        clusters.append(cluster)
        # assert len(cluster) == len(seed_words)
        # print (cluster)
        # print ()
    except Exception as e:
        print (e)
        continue

fw1.write(json.dumps(clusters, ensure_ascii=False))

fw.close()
fw1.close()



        
            