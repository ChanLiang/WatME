import nltk

# 下载必要的资源
nltk.download('averaged_perceptron_tagger')

# 对给定的单词进行词性标注
def get_word_pos(word):
    # pos = nltk.pos_tag([word])[0][1]
    pos = nltk.pos_tag([word])
    return pos

# 测试一下
print (get_word_pos("running")) # VBG
print (get_word_pos("run")) # VB
print (get_word_pos("address")) # NN
print (get_word_pos("addresses")) 
