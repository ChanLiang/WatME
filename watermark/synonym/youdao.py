import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet
import tqdm
from wordnet import get_exact_synonyms

model_dic = {}
with open('res_3c', 'r', encoding='utf-8') as r:
    for line in tqdm.tqdm(r):
        idx, word = line.strip().split()
        model_dic[word.strip()] = int(idx.strip())

# print (len(model_dic)) # 23239


def get_synonyms(i, word):
    # 检查单词是否存在多种词性
    st = set()
    for synset in wordnet.synsets(word):
        st.add(synset.pos())
    
    if len(st) > 1:
        # print(f"The word '{word}' has multiple parts of speech {st}. Synonyms cannot be determined.")
        return []

    word_pos = list(st)[0] if st else None
    # print (word_pos)

    url = f'https://www.youdao.com/w/eng/{word}/#keyfrom=dict2.index'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    synonyms = []
    try:
        pos_section = soup.findAll('span', {'class': 'pos'})
        #print (pos_section)
        # for pos in pos_section:
            # print (pos.text)
        if len(pos_section) > 1: # 多个词性;
            return []

        parts = soup.find('div', {'class': 'trans-container tab-content hide'})
        if not parts: # 没有同义词
            return []
        synonyms_section = parts.find_all('a', {'class': 'search-js'})
        for syn in synonyms_section: 
            sim_word = syn.text
            if sim_word == word:
                continue

            # 同义词和原词词性是否有重合
            if not word_pos or not wordnet.synsets(sim_word):
                synonyms.append(sim_word)
                break
            for synset in wordnet.synsets(sim_word):
                if synset.pos() == word_pos:      
                    synonyms.append(sim_word)
                    break
    except Exception as e:
        print(e)

    wordnet_res = get_exact_synonyms(word)

    res = list(set(synonyms) & set(wordnet_res))

    print (i, word, res, synonyms, wordnet_res)

    return res


print (get_synonyms(0, 'car'))
print (get_synonyms(0, 'hair'))
print (get_synonyms(0, 'Tencent'))
print (get_synonyms(0, 'like'))

# print (get_synonyms('he'))
# print (get_synonyms('is'))
# print (get_synonyms('or'))
# print (get_synonyms('id'))

# with open('res_3c', 'r', encoding='utf-8') as r, \
#     open('res_syn_3c', 'w', encoding='utf-8') as w:
#         # for line in tqdm.tqdm(r):
#         for i, line in enumerate(r):
#             idx, word = line.strip().split()
#             res = get_synonyms(i, word)
#             ret = [e.strip() for e in res if e.strip() in model_dic]
#             w.write(f'{idx}\t{word}\t{res}\t{ret}\n')
