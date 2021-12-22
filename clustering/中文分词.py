# encoding=utf-8
import jieba
import os


# 获取当前文件路径
print(os.getcwd())
# 获取上一级文件路径
print(os.path.dirname(os.getcwd()))
# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
f=open("/data/cuimengmeng/News_Event/data/老虎咬人事件/2007-03-23＠老虎咬人之后","r")
data=f.readlines()
print(data)
lines="".join(data)
seg_list=jieba.cut(lines,cut_all=False)
with open("result.txt", "w") as f:
    f.write("/ ".join(seg_list))
# print(type("/ ".join(seg_list)))
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
# 输出:
#
# 【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
#
# 【精确模式】: 我/ 来到/ 北京/ 清华大学
#
# 【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
#
# 【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学,深造