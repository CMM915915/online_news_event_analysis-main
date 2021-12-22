import jieba
from sklearn.cluster import KMeans
import re
from gensim.models import word2vec
import multiprocessing
import gensim
import numpy as np
import pandas as pd
import collections
import pandas
# mydict = ["result.txt"]
# file_path = '/data/cuimengmeng/News_Event/clustering/result2.txt'
# # 默认是精确模式
# test = WordCut()
# test.addDictionary(mydict)  # 加载自定义词典
# # 分词，去停用词（集成在类中了），不显示在console，保存分词后的文件到file_path目录
# test.seg_file(file_path, show=False, write=True)





# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./data/老虎咬人事件/2007-03-23＠老虎咬人之后', encoding='UTF-8').readlines()]
    return stopwords

def segment_text(source_corpus, train_corpus, coding, punctuation):
    '''
    切词,去除标点符号
    :param source_corpus: 原始语料
    :param train_corpus: 切词语料
    :param coding: 文件编码
    :param punctuation: 去除的标点符号
    :return:
    '''
    with open(source_corpus, 'r', encoding=coding) as f, open(train_corpus, 'w', encoding=coding) as w:
        for line in f:
            # 去除标点符号
            line = re.sub('[{0}]+'.format(punctuation), '', line.strip())
            # 切词
            words = jieba.cut(line)
            w.write(' '.join(words))

# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    # # # 默认是精确模式
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# 严格限制标点符号
strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
# 简单限制标点符号
simple_punctuation = '’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# 去除标点符号
punctuation = simple_punctuation + strict_punctuation

f = open("./data/老虎咬人事件/2007-03-23＠老虎咬人之后","r",encoding='UTF-8')
line = f.readlines()
lines = ''.join(line)

# with open("./data/result2.txt","w",encoding='UTF-8') as f:
#    f.write(seg_depart(lines))
# 是每个词的向量维度
size = 10
# 是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词
window = 5
# 设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃
min_count = 1
# 是训练的进程数，默认是当前运行机器的处理器核数。
workers = multiprocessing.cpu_count()
# 切词语料
train_corpus_text = './data/result2.txt'
# w2v模型文件
model_text = 'w2v_size_{0}.model'.format(size)

source_corpus = "./data/老虎咬人事件/2007-03-23＠老虎咬人之后"

coding = 'utf-8'

# 切词 @TODO 切词后注释
segment_text(source_corpus, train_corpus_text, coding, punctuation)



# w2v训练模型 @TODO 训练后注释
sentences = word2vec.Text8Corpus(train_corpus_text)   # 加载语料
model = word2vec.Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers)
model.save(model_text)

# 加载模型
model = gensim.models.Word2Vec.load(model_text)

g = open("./data/result2.txt", "r",encoding='UTF-8')  # 设置文件对象
std = g.read()  # 将txt文件的所有内容读入到字符串str中
g.close()  # 将文件关闭

cc = std.split(' ')  # ['标题', ':', '老虎', '咬人', '之后', '\n', '发布', '时间', ':', '2007', '-', '03




dd = []
kkl = dict()


'''
将每个词语向量化，并且append 在dd中，形成一个二维数组
并形成一个字典，index是序号，值是汉字
'''
index1 = []
for p in range(len(cc)):  # 3896
    hk = cc[p]
    if hk in model:
        vec = list(model.wv[hk])
        dd.append(vec)
        kkl[p] = hk
        index1.append(p)

# kkl {0: '标题', 1: ':', 2: '老虎', 3: '咬人', 4: '之后', 6: '发布', 7: '时间', 8: ':', 9: '2007', 10: '-', 11: '03', 12: '-'
#dd  [[0.021426199, -0.015310322, 0.028066937, 0.017086413, 0.020035753, -0.035560098, -0.042497594, -0.036129046, -0.0043878118, -0.026238237],

# 将二维数组转化成numpy
dd1 = np.array(dd)

estimator = KMeans(n_clusters=100)  # 构造聚类器   生成100个分类
estimator.fit(dd1)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签

# label_pred   [13 86 12 ...  3 36 48]    len:3746   100个种类




# index 是某条向量的序号，值是分类号
index1 = list(range(len(dd1)))    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,

vc = pd.Series(label_pred, index=index1)  # 转为一维数组，聚类标签和index1对应，index1为下标

aa = collections.Counter(label_pred)  # 统计器   Counter({5: 281, 9: 228, 0: 169, 6: 119, 17: 57, 19: 54, 40: 53, 16: 53, 20: 51, 55: 49, 13: 48,

v = pd.Series(aa)   # 使得下标为key，值为数组的值

v1 = v.sort_values(ascending=False)   # 按值从大到小排序


# 输出前10个类
for n in range(10):
    vc1 = vc[vc == v1.index[n]]  # 取前每个类别存到vc1


    vindex = list(vc1.index)

    kkp = pd.Series(kkl)  # len 3746


    print('第', n + 1, '类的前10个数据')


    ffg = kkp[vindex][:10]
    ffg1 = list(set(ffg))
    print(ffg1)