import json

def load_synonym_dict(path):
    with open(path, 'r', encoding='utf-8') as fr:
        syn_dict = json.load(fr)
    syn_set = set()
    for line in syn_dict:
        for cluster in line:
            for pair in cluster:
                syn_set.add(pair[1])
    return syn_dict, syn_set

synonym_dict, synonym_set = load_synonym_dict('synonym_clusters.json')
print (len(synonym_dict), len(synonym_set)) # 9838 15289

# 过滤掉完全相同的line;
with open('synonym_clusters_1.json', 'w', encoding='utf-8') as fw:
    line_set = set()
    new_dict = []
    for line in synonym_dict:
        cur_line = [e[1] for cluster in line for e in cluster]
        # print (cur_line)
        if sum(cur_line) not in line_set:
            new_dict.append(line)
            line_set.add(sum(cur_line))

    print (len(line_set), len(new_dict)) # 5443
    fw.write(json.dumps(new_dict, ensure_ascii=False))

print (len([e[1] for line in new_dict for cluster in line for e in cluster])) # 59581
print (len(set([e[1] for line in new_dict for cluster in line for e in cluster]))) # 15273, 新词典重复4次


# 如果一个line包含在了另一个line中，那么就删除这个line
with open('synonym_clusters_2.json', 'w', encoding='utf-8') as fw, \
    open('synonym_clusters_2_ids.json', 'w', encoding='utf-8') as fw1:
    line_set = set()
    line_set1 = []
    line_set2 = []
    new_dict = []
    new_dict1 = []
    for line in synonym_dict:
        cur_line = [e[1] for cluster in line for e in cluster]
        cur_line_sum = []
        for cluster in line:
            cur_line_sum.append(sum([e[1] for e in cluster]))

        flag = False
        for line1, line_sum in zip(line_set1, line_set2):
            if set(cur_line).issubset(set(line1)) or (len(set(cur_line) | set(line1)) - len(cur_line) < 2) or (len(set(cur_line_sum) | set(line_sum)) - len(cur_line_sum) < 2):
                flag = True
                break
        if flag:
            continue
        if sum(cur_line) not in line_set:
            new_dict.append(line)
            new_dict1.append([[pair[1] for pair in cluster] for cluster in line])
            line_set.add(sum(cur_line))
            line_set1.append(set(cur_line))
            line_set2.append(cur_line_sum)

    print (len(line_set), len(new_dict)) # 3384
    fw.write(json.dumps(new_dict, ensure_ascii=False))
    fw1.write(json.dumps(new_dict1, ensure_ascii=False))

print (len([e[1] for line in new_dict for cluster in line for e in cluster])) # 38407
print (len(set([e[1] for line in new_dict for cluster in line for e in cluster]))) # 15196, 新词典重复2.5次 


'''
2749 2749
27895
14691, 新词典重复1.9次
'''