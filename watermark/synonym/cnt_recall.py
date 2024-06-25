import sys

f = sys.argv[1]
res, cnt, st = 0, 0, set()
with open(f, 'r', encoding='utf-8') as r:
    for line in r:
        syn_list = eval(line.strip().split('\t')[-1])
        word = line.strip().split('\t')[1]
        if syn_list:
            res += 1
            st = st | set(syn_list) | set([word])
        cnt += 1

print (res, len(st), cnt) 
# 250 275 3229
# 14884 7838 23239