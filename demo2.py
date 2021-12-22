import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cluster import DBSCAN

corpus = []
flag = True
titles = []
with open("/data/cuimengmeng/News_Event/result.txt",encoding="utf-8") as f:

    index= 0
    for new in f.read().splitlines():
        if new.startswith("["):
            pass
        else:
            if new != "":
                corpus.append(new)


corpus = list(set(corpus))  # 369
# === 读文本分词和标记 === #
part_of_speech = []
word_after_cut = []
cut_corpus_iter = corpus.copy()
cut_corpus = corpus.copy()
for i in range(len(corpus)):
    cut_corpus_iter[i] = pseg.cut(corpus[i])
    cut_corpus[i] = ""
    for every in cut_corpus_iter[i]:
        cut_corpus[i] = (cut_corpus[i] + " " + str(every.word)).strip()
        part_of_speech.append(every.flag)
        word_after_cut.append(every.word)
word_pos_dict = {word_after_cut[i]: part_of_speech[i] for i in range(len(word_after_cut))}

# === 提取tf-idf权值 === #
Count_vectorizer = CountVectorizer()
transformer = TfidfTransformer()  # 用于统计每个词语的tf-idf权值
tf_idf = transformer.fit_transform(Count_vectorizer.fit_transform(cut_corpus))
# （5316，2039）第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
word = Count_vectorizer.get_feature_names()  # ，获取词袋模型中的所有词语
weight = tf_idf.toarray()  # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重

# === 得到标记的新权重 === #
word_weight = [1 for i in range(len(word))]
for i in range(len(word)):
    if word[i] not in word_pos_dict.keys():
        continue
    if word_pos_dict[word[i]] == 'n':
        word_weight[i] = 1.2
    elif word_pos_dict[word[i]] == "vn":
        word_weight[i] = 1.1
    elif word_pos_dict[word[i]] == "m":
        word_weight[i] = 0
    else:  # 权重调整可以根据实际情况进行更改
        continue
word_weight = np.array(word_weight)
new_weight = weight.copy()
for i in range(len(weight)):
    for j in range(len(word)):
        new_weight[i][j] = weight[i][j] * word_weight[j]


# === 固定DBSCAN模型 得到类标签 === #

DBS_clf = DBSCAN(eps=1, min_samples=4)
DBS_clf.fit(new_weight)
print(DBS_clf.labels_)

# 根据标签定义类的功能和原始预料
labels = DBS_clf.labels_
for index in range(len(labels)):
    print("----------{0}".format(labels[index] + 1))
    print(corpus[index])


