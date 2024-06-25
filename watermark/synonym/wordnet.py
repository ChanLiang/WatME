import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

def get_exact_synonyms(word):
    synonyms = []
    # 查找与输入单词相关联的所有同义词集：
    synsets = wordnet.synsets(word)
    # print (synsets)

    # 对于每个同义词集，使用lemmas方法获取其中的所有单词，并将这些单词添加到同义词列表中
    for synset in synsets:
        # print (synset, synset.lemmas())
        for lemma in synset.lemmas():
            cur = lemma.name().lower()
            if '_' in cur or word.lower() == cur:  
                continue
            synonyms.append(cur)
    synonyms = list(set(synonyms))

    # print ('--'*20)
    return synonyms

# print (get_exact_synonyms('car')) # ['automobile', 'cable_car', 'machine', 'railcar', 'railroad_car', 'gondola', 'auto', 'elevator_car', 'motorcar', 'railway_car']
# print (get_exact_synonyms('Tencent')) # []
# print (get_exact_synonyms('hair')) # ['tomentum', "hair's-breadth", 'hairsbreadth', 'haircloth', 'whisker', 'fuzz', 'pilus']
# print (get_exact_synonyms('like')) # ['corresponding', 'same', 'wish', 'ilk', 'care', 'similar', 'alike', 'the_like', 'comparable', 'the_likes_of']
# print (get_exact_synonyms('he')) # ['corresponding', 'same', 'wish', 'ilk', 'care', 'similar', 'alike', 'the_like', 'comparable', 'the_likes_of']
# print ()

# get_exact_synonyms('car')
# get_exact_synonyms('hair')
# get_exact_synonyms('like')