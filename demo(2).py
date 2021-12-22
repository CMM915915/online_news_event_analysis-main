# -*- coding:utf-8 -*-
import jieba.analyse
import re

# 读取数据
news_list = []
titles = []
# with open("./news.txt",encoding="utf-8") as f:
#
#     index= 0
#     for new in f.read().splitlines():
#         if new.startswith("["):
#             new1 = new.split("\"")[-2]
#             if new1 != "":
#                 titles.append(new1)
#         else:
#             if new != "":
#                 news_list.append(new)
context = []
flag =True
with open("./news.txt",encoding="utf-8") as f:

    index= 0
    for new in f.read().splitlines():
        index += 1
        if new.startswith("["):
            new1 = new.split("\"")[-2]
            if new1 != "":
                news_list.append(new1)
            else:
                flag = False
        else:
            if flag:
                if new != "":
                    context.append(new)
                else:
                    news_list.pop()
            else:
                flag = True
print(index)
# 830

title_context_dict = {} # 去重
for new_index in range(len(news_list)):
    title_context_dict[news_list[new_index]] = context[new_index]
news_list = []
context = []
for k,v in title_context_dict.items():
    news_list.append(k)
    context.append(v)


news_list = context
print(len(news_list)) # 362


keywords_list =[]
# 提取每个新闻前十的关键词
for index in range(len(news_list)):
    keywords = jieba.analyse.extract_tags(news_list[index], topK=10, withWeight=False, allowPOS=([]))
    keywords_list.append(keywords)



type = 0
type_new_list = [[type,keywords_list[0],[0]]]   # 记录了新闻和类


for index in range(1,len(keywords_list)):
    close_list = []
    for i in type_new_list:
        close_keywords = list(set(i[1]).intersection(keywords_list[index]))   # 判断是否存在交集
        close_list.append(len(close_keywords))

    type_max_index = 0
    type_max = 0
    for tindex in range(0,len(close_list)):  # 找最大的type
        if type_max_index < close_list[tindex]:
            type_max_index = tindex
            type_max = close_list[type_max_index]

    if type_max == 0:  # 创建一个新类
        type += 1
        type_new_list.append([type,keywords_list[index],[index]])
    else:
        type_new_list[type_max_index][-1].append(index)



#
for data in type_new_list:
    print("--------第{0}类-----------".format(data[0]))
    for new in data[2]:

        print(news_list[new])


# news_types_dict = {}
# for data in type_new_list:
#     for d in data[2]:
#         news_types_dict[context[d]] = data[0]
#
#
# for k,v in news_types_dict.items():
#     print("{0}-----------".format(v))
#
#     print(k)










