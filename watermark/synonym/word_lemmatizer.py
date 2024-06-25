
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer



# 创建WordNetLemmatizer对象
wnl = WordNetLemmatizer()

# 定义一个函数，输入一个单词并返回其原型
def get_lemma(word):
    lemma = wnl.lemmatize(word)  # 默认还原为动词原型
    return lemma

# 测试一下
print(get_lemma('running'))  # 输出：run
print(get_lemma('ran'))  # 输出：run
print(get_lemma('wolves'))  # 输出：wolf
print(get_lemma('goose'))  # 输出：goose（原型和单数形式相同的特殊情况）
print(get_lemma('easiest'))
print(get_lemma('better'))


# from pattern.en import lemma
# print (lemma('running'))
# print (lemma('ran'))
# print (lemma('wolves'))
# print (lemma('goose'))
# print (lemma('easiest'))
# print (lemma('better'))