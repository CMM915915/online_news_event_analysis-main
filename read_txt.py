import os
import re
ss='["aaa",""新“视”窗点亮精彩生活（走向我们的小康生活）"]         '

l=ss.split(",")[1].rstrip(" ").rstrip("']'")

with open("result.txt","r") as f:
    print(f.readline().split(",")[1].split(" ")[-1])
    l=f.readline().split(",")[0]
    res=re.findall(r'(?<=[).*?(?=])',l)
    # print(l)
