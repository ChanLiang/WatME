import unicodedata

def normalize(token):
    """Normalize token to NFC and remove non-letter characters"""
    token = unicodedata.normalize("NFC", token) # 将 token 规范化为 NFC 编码
    token = ''.join(c for c in token if unicodedata.category(c)[0] in ('L', 'M', 'N')) # 去除 token 中的特殊字符
    return token


print (normalize('\u010e'))
print (normalize('\u0120up'))
print (normalize('\u00e2'))

print ('\u0120up'.strip('\u0120'))