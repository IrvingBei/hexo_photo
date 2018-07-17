---
title: 【NLTK基础教程】01-02 利用nltk统计词频
date: 2018-01-21 20:26:59
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/nltk.jpg
tags: NLTK基础教程
categories: [NLP,NLTK]
---
在上篇中，简单的介绍了三种获取有效文本的方法，那么接下来就利用nltk来统计这些文本中出现的次数。
<!--more-->
我们首先来看下传统统计词频的方法：

```
import operator
freq_dis={}
for tok in tokens:
    if tok in freq_dis:
        freq_dis[tok]+=1
    else:
        freq_dis[tok]=1

sorted_freq_dist=sorted(freq_dis.items(),key=operator.itemgetter(1),reverse=True)
print(sorted_freq_dist[:25])
```
统计结果如下：

```
[('Python', 59), ('>>>', 24), ('the', 21), ('and', 21), ('to', 17), ('is', 17), ('of', 17), ('=', 14), ('for', 11), ('News', 11), ('Events', 11), ('a', 10), ('#', 9), ('More', 9), ('3', 8), ('in', 8), ('with', 7), ('Community', 7), ('...', 7), ('Docs', 6), ('Guide', 6), ('Software', 6), ('The', 5), ('1', 5), ('that', 5)]
```
利用nltk来统计文本词频如下：

```
import nltk
Freq_dist_nltk=nltk.FreqDist(tokens)
print(Freq_dist_nltk)
for k,v in Freq_dist_nltk.items():
    print(str(k)+":"+str(v))
Freq_dist_nltk.plot(50,cumulative=False)
```
相比之下，利用nltk库来实现，确实便利了很多。
![image](http://img.blog.csdn.net/20180121203411270?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveHl6MTU4NDE3MjgwOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
好吧，《NLTK基础教程》第一章基本上就结束了，这一章主要是简单介绍了python的语法，然后引出NLTK。
