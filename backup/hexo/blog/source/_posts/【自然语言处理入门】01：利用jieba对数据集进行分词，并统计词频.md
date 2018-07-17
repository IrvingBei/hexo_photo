---
title: 【自然语言处理入门】01：利用jieba对数据集进行分词，并统计词频
date: 2017-12-13 09:20:00
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/nlp2.jpg
tags: 自然语言处理
categories: [NLP,MDCourse]
---

### 一、基本要求
使用jieba对垃圾短信数据集进行分词，然后统计其中的单词出现的个数，找到出现频次最高的top100个词。
<!--more-->
### 二、完整代码
```python
# -*- coding: UTF-8 -*-
from collections import Counter
import jieba.analyse
import re
import time

#分词模板
def cut_word(datapath):
    with open(datapath, 'r',encoding='utf-8') as fr:
        string=fr.read()
        print(type(string))
        #对文件中的非法字符进行过滤
        data=re.sub(r"[\s+\.\!\/_,$%^*(【】：\]\[\-:;+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[0-9]+","",string)
        word_list= jieba.cut(data)
        print(word_list)
        return word_list
#词频统计模块
def statistic_top_word(word_list,top=100):
    #统计每个单词出现的次数，别将结果转化为键值对（即字典）
    result= dict(Counter(word_list))
    print(result)
    #sorted对可迭代对象进行排序
    #items()方法将字典的元素转化为了元组，而这里key参数对应的lambda表达式的意思则是选取元组中的第二个元素作为比较参数
    #排序厚的结果是一个列表，列表中的每个元素是一个将原字典中的键值对转化为的元祖
    sortlist=sorted(result.items(),key=lambda item:item[1],reverse=True)
    resultlist=[]
    for i in range(0,top):
        resultlist.append(sortlist[i])
    return resultlist

#主函数
def main():
    #设置数据集地址
    datapath='F:\\python3\\nlp\\data\\spam.txt'
    #对文本进行分词
    word_list=cut_word(datapath)
    #统计文本中的词频
    statistic_result=statistic_top_word(word_list,100)
    #输出统计结果
    print(statistic_result)

if __name__ == "__main__":
    main()
```

### 三、相关知识点
- 1、jieba分词：三种模式，详见[相关介绍](http://www.jianshu.com/p/c434be968dee)
- 2、对字典进行排序：字典可以实现对键和值分别排序。详见[原文链接](http://blog.csdn.net/tangtanghao511/article/details/47810729)
- 3、python 过滤中文、英文标点特殊符号：在进行分词前，主要是利用正则表达式对欲分词文本进行过滤，利用re.sub（）函数对“非法”字符进行空字符替换。详见[原文链接](http://blog.csdn.net/mach_learn/article/details/41744487)

### 四、相关参考
- 	1、[python数据分析：jieba模块对数据进行切词并统计出现每个词的次数](http://www.linuxyw.com/810.html)
- 	2、[python的sorted函数对字典按key排序和按value排序](http://blog.csdn.net/tangtanghao511/article/details/47810729)
- 	3、[python 过滤中文、英文标点特殊符号](http://blog.csdn.net/mach_learn/article/details/41744487)