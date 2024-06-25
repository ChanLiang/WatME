from transformers import AutoTokenizer, AutoModel
import torch

# 加载BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 输入两个词
word1 = "dog"
word2 = "cat"

# 将词转换为BERT输入格式
encoded_dict = tokenizer.encode_plus(
                        word1, 
                        word2,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt')

# 获取BERT模型的输出
with torch.no_grad():
    outputs = model(**encoded_dict)
print (outputs)

# 获取CLS的输出向量
vecs = outputs[1]

# 计算余弦相似度
cos_sim = torch.nn.functional.cosine_similarity(vecs[0], vecs[1], dim=0)

print(f"相似度: {cos_sim}")

