
cnt = 0
with open('synonym_clusters.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    all_ids = set([pair[1] for line in lines for cluster in line for pair in cluster])
    cnt = len(set(all_ids))
    print (list(all_ids)[:10])
    # cnt1 = len(list([pair[1] for line in lines for cluster in line for pair in cluster]))
    # print (len(lines), cnt1, cnt, cnt/50257) # 9838 【115189】 【15289】 0.304
    print (len(lines), cnt, cnt/50257) # 9838 【15289】 0.304

    '''
    一共有1.5w个词, 但是一共出现了11w多个词。。。
    7/8的词是重复的,会被重复划分到绿队：(1)对称性的词;(2)一词多义;
    但其实多划分没什么，绿对越大效果越好。同义词划分的初衷是不希望一个簇中的所有词都被划分到红队了（特别是比较小的簇），这样就会失去一个簇的语义。
    '''