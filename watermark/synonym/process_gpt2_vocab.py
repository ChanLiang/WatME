import json
import unicodedata
import nltk
import string
from nltk.corpus import wordnet
from nltk.corpus import words
import tqdm

word_set = set(words.words())
punc_set = set(list(string.punctuation))
spec_set = set(["$", "£", "€", "©", "™", "®", "±", "§", "¶", "<|endoftext|>"])

def normalize(token):
    """Normalize token to NFC and remove non-letter characters"""
    token = unicodedata.normalize("NFC", token) # 将 token 规范化为 NFC 编码
    token = ''.join(c for c in token if unicodedata.category(c)[0] in ('L', 'M', 'N')) # 去除 token 中的特殊字符
    return token

def is_valid_english_word(word):
    # if word in punc_set or word in spec_set:
    #     return True
    word = normalize(word.lstrip("Ġ")).lower().strip()
    if word and (wordnet.synsets(word) or word in word_set):  # 如果在词典中，但是可能是大写字母开头，再检查一遍
        return True
    else:
        return False

with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Apply normalization to all tokens in vocab.json, and remove Ġ prefix
# vocab_normalized = {normalize(token.lstrip("Ġ")): index for token, index in vocab.items()}

# Print the normalized vocabulary
for token, index in tqdm.tqdm(vocab.items()):
    # if len(token) >= 3 and wordnet.synsets(token) != [] and is_valid_english_word(token):
    if is_valid_english_word(token):
        token = normalize(token.lstrip("Ġ"))
        print(f"{index}\t{token}")
        # print (token.strip())
