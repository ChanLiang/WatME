import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet
import tqdm
from wordnet import get_exact_synonyms

def get_pos(word):
    url = f'https://www.youdao.com/w/eng/{word}/#keyfrom=dict2.index'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    try:
        pos_section = [e.text for e in soup.findAll('span', {'class': 'pos'})] # 词性
        # print (pos_section)
        return pos_section
    except Exception as e:
        print(e)
    return []


def get_synonyms(i=0, word='car'):
    url = f'https://www.youdao.com/w/eng/{word}/#keyfrom=dict2.index'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    synonyms = []
    try:
        pos_section = [e.text for e in soup.findAll('span', {'class': 'pos'})] # 词性
        # print (pos_section)
        # if len(pos_section) > 1: # 多个词性
        #     return []
        parts = soup.find('div', {'class': 'trans-container tab-content hide'})
        if not parts: # 没有同义词
            return []
        synonyms_section = parts.find_all('a', {'class': 'search-js'})
        for syn in synonyms_section: 
            sim_word = syn.text.strip()
            if sim_word == word:
                continue
            sim_pos_section = get_pos(sim_word)
            # if len(set(pos_section) & set(sim_pos_section)) == 0: # 词性不同
            #     continue
            synonyms.append(sim_word)

    except Exception as e:
        print(e)

    return synonyms


# print (get_synonyms(word='car')) # ['automobile', 'motor']
# print (get_synonyms(word='cars')) # []
# print (get_synonyms(word='hair')) # ['capello', 'poil', 'pilar']
# print (get_synonyms(word='Tencent')) # []
# print (get_synonyms(word='like')) # ['love', 'think', 'prefer', 'affect', 'as per', 'similar', 'alike', 'interest', 'bent', 'fond of', 'perhaps', 'maybe', 'supposedly', 'on the cards', 'forse']

# print (get_synonyms(word='he'))  # 排除
# print (get_synonyms(word='is'))  # 排除
# print (get_synonyms(word='or')) # ['either', 'ossia']
# print (get_synonyms(word='id'))

# model_dic = {} # token to idx
# # with open('res_3c', 'r', encoding='utf-8') as r:
# with open('id_token_2c2', 'r', encoding='utf-8') as r:
#     for line in tqdm.tqdm(r):
#         idx, word = line.strip().split()
#         model_dic[word.strip()] = int(idx.strip())

# # print (len(model_dic)) # 23239
# # print (len(model_dic)) # 35005

# # # with open('res_3c', 'r', encoding='utf-8') as r, \
#     # open('res_syn_3c_only_yd_pos_filter', 'w', encoding='utf-8') as w:
# with open('id_token_2c2', 'r', encoding='utf-8') as r, \
#     open('id_token_2c2_synonym', 'w', encoding='utf-8') as w:
#         # for line in tqdm.tqdm(r):
#         for i, line in tqdm.tqdm(enumerate(r)):
#             idx, word = line.strip().split()
#             syn_list = get_synonyms(i, word)
#             syn_in_dict = [e.strip() for e in syn_list if e.strip() in model_dic]
#             print (word, syn_list, syn_in_dict)
#             w.write(f'{idx}\t{word}\t{all_syn_list}\t{syn_in_dict}\n')
