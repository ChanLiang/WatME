import pickle

synonyms_file = "output/llama2_supervised_synonym_mining/id_token_synonyms_list_2c_youdao.pkl"
print("Loading previously inferred synonyms...")
with open(synonyms_file, 'rb') as f:
    id_token_synonyms_list = pickle.load(f)

# 11875 3 [262, 'in', ['within', 'inside', 'modern', 'fashionable', 'tony']]
print (len(id_token_synonyms_list), len(id_token_synonyms_list[0]), id_token_synonyms_list[0])