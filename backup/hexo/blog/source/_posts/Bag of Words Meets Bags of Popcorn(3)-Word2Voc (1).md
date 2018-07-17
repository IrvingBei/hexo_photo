---
title: Bag of Words Meets Bags of Popcorn(3)-Word2Voc
date: 2018-6-14 17:27:51
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/bag.png
tags: 
      - Kaggle
      - 文本分类
categories: [Kaggle,nlp]
---

词带模型：[Bag of Words Meets Bags of Popcorn(1)-Bag of Words](http://bei.dreamcykj.com/2018/06/13/Bag%20of%20Words%20Meets%20Bags%20of%20Popcorn%281%29-Bag%20of%20Words/)
Tfidf模型：[Bag of Words Meets Bags of Popcorn(2)-tfidf](http://bei.dreamcykj.com/2018/06/14/Bag%20of%20Words%20Meets%20Bags%20of%20Popcorn%282%29-tfidf%20%281%29/)
这一节采用词向量
<!--more-->
#### 1、读取数据


```python
import pandas as pd
train=pd.read_csv('./data/labeledTrainData.tsv',header=0,delimiter="\t",quoting=3)
test=pd.read_csv("./data/testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train=pd.read_csv("./data/unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
# 查看数据的大小
print("read %d labeled train reviews,%d labeled test reviews and %d unlabeled reviews\n"%(train["review"].size,len(test),len(unlabeled_train)))
```

    read 25000 labeled train reviews,25000 labeled test reviews and 50000 unlabeled reviews
    


#### 2、数据清洗

##### 2、1 对每一句评论进行数据清洗


```python
from bs4 import BeautifulSoup
import re 
from nltk.corpus import stopwords

def review_to_wordlist(review,remove_stopwords=False):
    # 去掉html标签
    review_text=BeautifulSoup(review).get_text()
    # 去掉标点符号和非法字符
    review_text=re.sub("[^a-zA-Z]"," ",review_text)
    # 全部转化为小写,并以空格分割
    words=review_text.lower().split()
    # 去停用词
    if remove_stopwords:
        stops=set(stopwords.words("english"))
        words=[w for w in words if w not in stops]
    return words
```

##### 2.2 对每一篇评论进行数据清洗
将评论段落转换为句子，返回句子列表，每个句子由一堆词组成


```python
import nltk.data
tokenizer=nltk.data.load("tokenizers/punkt/english.pickle")
def review_to_sentences(review,tokenizer,remove_stopwords=False):
    # 将评论按句子分割
    raw_sentences=tokenizer.tokenize(review.strip())
    # 对一个评论的每一句话进行清洗
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            # 调用数据清洗方法，对该句评论进行清洗
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    # 返回清洗后的这一篇评论
    return sentences   
```

#### 3、构造语料集
语料集中的数据包括已标注好的训练集和未标注好的训练集


```python
sentences=[] # 装有所有评论数据的一个list，相当于一个语料集,格式为：[[a.b.c].[a.b.c.d]...]
print("Parsing sentences from training set")
for review in train["review"]:
    sentences+=review_to_sentences(review,tokenizer)

print("Parsing sentence from unlabeled set")
for review in unlabeled_train["review"]:
    sentences+=review_to_sentences(review,tokenizer)
# 查看语料集的大小
print(len(sentences))
```

    Parsing sentences from training set
    795538



```python
# 查看部分数据
print(sentences[0])
print(sentences[1])
```

    ['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'started', 'listening', 'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary', 'here', 'and', 'there', 'watched', 'the', 'wiz', 'and', 'watched', 'moonwalker', 'again']
    ['maybe', 'i', 'just', 'want', 'to', 'get', 'a', 'certain', 'insight', 'into', 'this', 'guy', 'who', 'i', 'thought', 'was', 'really', 'cool', 'in', 'the', 'eighties', 'just', 'to', 'maybe', 'make', 'up', 'my', 'mind', 'whether', 'he', 'is', 'guilty', 'or', 'innocent']


#### 4、构建word2vec模型
这里主要使用gensim里的word2vec，相关介绍和使用情况见：python︱gensim训练word2vec及相关函数与功能理解 https://blog.csdn.net/sinat_26917383/article/details/69803018

##### 4、1训练模型


```python
import gensim,logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)

num_features=300
min_word_count=40
num_workers=4
context=10
downsampling=1e-3
print("Training model...")
model=gensim.models.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
model.init_sims(replace=True)
model_name="300features_40minwords_10context"
model.save("./data/"+model_name)
```

    2018-06-14 16:17:04,394 : INFO : collecting all words and their counts
    2018-06-14 16:17:04,394 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2018-06-14 16:17:04,439 : INFO : PROGRESS: at sentence #10000, processed 225803 words, keeping 17776 word types
    2018-06-14 16:17:04,486 : INFO : PROGRESS: at sentence #20000, processed 451892 words, keeping 24948 word types
    2018-06-14 16:17:04,530 : INFO : PROGRESS: at sentence #30000, processed 671315 words, keeping 30034 word types
    2018-06-14 16:17:04,576 : INFO : PROGRESS: at sentence #40000, processed 897815 words, keeping 34348 word types


    Training model...


    2018-06-14 16:17:04,621 : INFO : PROGRESS: at sentence #50000, processed 1116963 words, keeping 37761 word types
    2018-06-14 16:17:04,666 : INFO : PROGRESS: at sentence #60000, processed 1338404 words, keeping 40723 word types
    2018-06-14 16:17:04,711 : INFO : PROGRESS: at sentence #70000, processed 1561580 words, keeping 43333 word types
    2018-06-14 16:17:04,759 : INFO : PROGRESS: at sentence #80000, processed 1780887 words, keeping 45714 word types
    2018-06-14 16:17:04,804 : INFO : PROGRESS: at sentence #90000, processed 2004996 words, keeping 48135 word types
    2018-06-14 16:17:04,849 : INFO : PROGRESS: at sentence #100000, processed 2226966 words, keeping 50207 word types
    2018-06-14 16:17:04,894 : INFO : PROGRESS: at sentence #110000, processed 2446580 words, keeping 52081 word types
    2018-06-14 16:17:04,939 : INFO : PROGRESS: at sentence #120000, processed 2668775 words, keeping 54119 word types
    2018-06-14 16:17:04,985 : INFO : PROGRESS: at sentence #130000, processed 2894303 words, keeping 55847 word types
    2018-06-14 16:17:05,029 : INFO : PROGRESS: at sentence #140000, processed 3107005 words, keeping 57346 word types
    2018-06-14 16:17:05,075 : INFO : PROGRESS: at sentence #150000, processed 3332627 words, keeping 59055 word types
    2018-06-14 16:17:05,120 : INFO : PROGRESS: at sentence #160000, processed 3555315 words, keeping 60617 word types
    2018-06-14 16:17:05,166 : INFO : PROGRESS: at sentence #170000, processed 3778655 words, keeping 62077 word types
    2018-06-14 16:17:05,212 : INFO : PROGRESS: at sentence #180000, processed 3999236 words, keeping 63496 word types
    2018-06-14 16:17:05,258 : INFO : PROGRESS: at sentence #190000, processed 4224449 words, keeping 64794 word types
    2018-06-14 16:17:05,304 : INFO : PROGRESS: at sentence #200000, processed 4448603 words, keeping 66087 word types
    2018-06-14 16:17:05,350 : INFO : PROGRESS: at sentence #210000, processed 4669967 words, keeping 67390 word types
    2018-06-14 16:17:05,397 : INFO : PROGRESS: at sentence #220000, processed 4894968 words, keeping 68697 word types
    2018-06-14 16:17:05,443 : INFO : PROGRESS: at sentence #230000, processed 5117545 words, keeping 69958 word types
    2018-06-14 16:17:05,490 : INFO : PROGRESS: at sentence #240000, processed 5345050 words, keeping 71167 word types
    2018-06-14 16:17:05,535 : INFO : PROGRESS: at sentence #250000, processed 5559165 words, keeping 72351 word types
    2018-06-14 16:17:05,581 : INFO : PROGRESS: at sentence #260000, processed 5779146 words, keeping 73478 word types
    2018-06-14 16:17:05,627 : INFO : PROGRESS: at sentence #270000, processed 6000435 words, keeping 74767 word types
    2018-06-14 16:17:05,674 : INFO : PROGRESS: at sentence #280000, processed 6226314 words, keeping 76369 word types
    2018-06-14 16:17:05,720 : INFO : PROGRESS: at sentence #290000, processed 6449474 words, keeping 77839 word types
    2018-06-14 16:17:05,767 : INFO : PROGRESS: at sentence #300000, processed 6674077 words, keeping 79171 word types
    2018-06-14 16:17:05,814 : INFO : PROGRESS: at sentence #310000, processed 6899391 words, keeping 80480 word types
    2018-06-14 16:17:05,861 : INFO : PROGRESS: at sentence #320000, processed 7124278 words, keeping 81808 word types
    2018-06-14 16:17:05,908 : INFO : PROGRESS: at sentence #330000, processed 7346021 words, keeping 83030 word types
    2018-06-14 16:17:05,956 : INFO : PROGRESS: at sentence #340000, processed 7575533 words, keeping 84280 word types
    2018-06-14 16:17:06,002 : INFO : PROGRESS: at sentence #350000, processed 7798803 words, keeping 85425 word types
    2018-06-14 16:17:06,048 : INFO : PROGRESS: at sentence #360000, processed 8019427 words, keeping 86596 word types
    2018-06-14 16:17:06,100 : INFO : PROGRESS: at sentence #370000, processed 8246619 words, keeping 87708 word types
    2018-06-14 16:17:06,146 : INFO : PROGRESS: at sentence #380000, processed 8471766 words, keeping 88878 word types
    2018-06-14 16:17:06,194 : INFO : PROGRESS: at sentence #390000, processed 8701497 words, keeping 89907 word types
    2018-06-14 16:17:06,240 : INFO : PROGRESS: at sentence #400000, processed 8924446 words, keeping 90916 word types
    2018-06-14 16:17:06,286 : INFO : PROGRESS: at sentence #410000, processed 9145796 words, keeping 91880 word types
    2018-06-14 16:17:06,331 : INFO : PROGRESS: at sentence #420000, processed 9366876 words, keeping 92912 word types
    2018-06-14 16:17:06,378 : INFO : PROGRESS: at sentence #430000, processed 9594413 words, keeping 93932 word types
    2018-06-14 16:17:06,425 : INFO : PROGRESS: at sentence #440000, processed 9821166 words, keeping 94906 word types
    2018-06-14 16:17:06,471 : INFO : PROGRESS: at sentence #450000, processed 10044928 words, keeping 96036 word types
    2018-06-14 16:17:06,519 : INFO : PROGRESS: at sentence #460000, processed 10277688 words, keeping 97088 word types
    2018-06-14 16:17:06,566 : INFO : PROGRESS: at sentence #470000, processed 10505613 words, keeping 97933 word types
    2018-06-14 16:17:06,612 : INFO : PROGRESS: at sentence #480000, processed 10725997 words, keeping 98862 word types
    2018-06-14 16:17:06,659 : INFO : PROGRESS: at sentence #490000, processed 10952741 words, keeping 99871 word types
    2018-06-14 16:17:06,704 : INFO : PROGRESS: at sentence #500000, processed 11174397 words, keeping 100765 word types
    2018-06-14 16:17:06,752 : INFO : PROGRESS: at sentence #510000, processed 11399672 words, keeping 101699 word types
    2018-06-14 16:17:06,798 : INFO : PROGRESS: at sentence #520000, processed 11623020 words, keeping 102598 word types
    2018-06-14 16:17:06,845 : INFO : PROGRESS: at sentence #530000, processed 11847418 words, keeping 103400 word types
    2018-06-14 16:17:06,893 : INFO : PROGRESS: at sentence #540000, processed 12072033 words, keeping 104265 word types
    2018-06-14 16:17:06,942 : INFO : PROGRESS: at sentence #550000, processed 12297571 words, keeping 105133 word types
    2018-06-14 16:17:06,990 : INFO : PROGRESS: at sentence #560000, processed 12518861 words, keeping 105997 word types
    2018-06-14 16:17:07,040 : INFO : PROGRESS: at sentence #570000, processed 12747916 words, keeping 106787 word types
    2018-06-14 16:17:07,089 : INFO : PROGRESS: at sentence #580000, processed 12969412 words, keeping 107665 word types
    2018-06-14 16:17:07,138 : INFO : PROGRESS: at sentence #590000, processed 13194937 words, keeping 108501 word types
    2018-06-14 16:17:07,197 : INFO : PROGRESS: at sentence #600000, processed 13417135 words, keeping 109218 word types
    2018-06-14 16:17:07,245 : INFO : PROGRESS: at sentence #610000, processed 13638158 words, keeping 110092 word types
    2018-06-14 16:17:07,294 : INFO : PROGRESS: at sentence #620000, processed 13864483 words, keeping 110837 word types
    2018-06-14 16:17:07,343 : INFO : PROGRESS: at sentence #630000, processed 14088769 words, keeping 111610 word types
    2018-06-14 16:17:07,392 : INFO : PROGRESS: at sentence #640000, processed 14309552 words, keeping 112416 word types
    2018-06-14 16:17:07,441 : INFO : PROGRESS: at sentence #650000, processed 14535308 words, keeping 113196 word types
    2018-06-14 16:17:07,490 : INFO : PROGRESS: at sentence #660000, processed 14758098 words, keeping 113945 word types
    2018-06-14 16:17:07,538 : INFO : PROGRESS: at sentence #670000, processed 14981482 words, keeping 114643 word types
    2018-06-14 16:17:07,587 : INFO : PROGRESS: at sentence #680000, processed 15206314 words, keeping 115354 word types
    2018-06-14 16:17:07,636 : INFO : PROGRESS: at sentence #690000, processed 15428507 words, keeping 116131 word types
    2018-06-14 16:17:07,686 : INFO : PROGRESS: at sentence #700000, processed 15657213 words, keeping 116943 word types
    2018-06-14 16:17:07,732 : INFO : PROGRESS: at sentence #710000, processed 15880202 words, keeping 117596 word types
    2018-06-14 16:17:07,779 : INFO : PROGRESS: at sentence #720000, processed 16105489 words, keeping 118221 word types
    2018-06-14 16:17:07,826 : INFO : PROGRESS: at sentence #730000, processed 16331870 words, keeping 118954 word types
    2018-06-14 16:17:07,872 : INFO : PROGRESS: at sentence #740000, processed 16552903 words, keeping 119668 word types
    2018-06-14 16:17:07,918 : INFO : PROGRESS: at sentence #750000, processed 16771230 words, keeping 120295 word types
    2018-06-14 16:17:07,964 : INFO : PROGRESS: at sentence #760000, processed 16990622 words, keeping 120930 word types
    2018-06-14 16:17:08,012 : INFO : PROGRESS: at sentence #770000, processed 17217759 words, keeping 121703 word types
    2018-06-14 16:17:08,060 : INFO : PROGRESS: at sentence #780000, processed 17447905 words, keeping 122402 word types
    2018-06-14 16:17:08,108 : INFO : PROGRESS: at sentence #790000, processed 17674981 words, keeping 123066 word types
    2018-06-14 16:17:08,134 : INFO : collected 123504 word types from a corpus of 17798082 raw words and 795538 sentences
    2018-06-14 16:17:08,134 : INFO : Loading a fresh vocabulary
    2018-06-14 16:17:08,204 : INFO : min_count=40 retains 16490 unique words (13% of original 123504, drops 107014)
    2018-06-14 16:17:08,205 : INFO : min_count=40 leaves 17238940 word corpus (96% of original 17798082, drops 559142)
    2018-06-14 16:17:08,247 : INFO : deleting the raw counts dictionary of 123504 items
    2018-06-14 16:17:08,249 : INFO : sample=0.001 downsamples 48 most-common words
    2018-06-14 16:17:08,250 : INFO : downsampling leaves estimated 12749658 word corpus (74.0% of prior 17238940)
    2018-06-14 16:17:08,286 : INFO : estimated required memory for 16490 words and 300 dimensions: 47821000 bytes
    2018-06-14 16:17:08,286 : INFO : resetting layer weights
    2018-06-14 16:17:08,497 : INFO : training model with 4 workers on 16490 vocabulary and 300 features, using sg=0 hs=0 sample=0.001 negative=5 window=10
    2018-06-14 16:17:09,516 : INFO : EPOCH 1 - PROGRESS: at 6.40% examples, 812589 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:10,525 : INFO : EPOCH 1 - PROGRESS: at 13.20% examples, 832030 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:11,526 : INFO : EPOCH 1 - PROGRESS: at 20.16% examples, 847270 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:12,529 : INFO : EPOCH 1 - PROGRESS: at 27.14% examples, 856397 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:13,535 : INFO : EPOCH 1 - PROGRESS: at 34.33% examples, 865561 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:14,549 : INFO : EPOCH 1 - PROGRESS: at 41.21% examples, 865831 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:15,556 : INFO : EPOCH 1 - PROGRESS: at 48.00% examples, 866028 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:16,560 : INFO : EPOCH 1 - PROGRESS: at 55.16% examples, 871678 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:17,561 : INFO : EPOCH 1 - PROGRESS: at 62.03% examples, 873356 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:18,569 : INFO : EPOCH 1 - PROGRESS: at 70.05% examples, 887453 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:19,574 : INFO : EPOCH 1 - PROGRESS: at 78.55% examples, 905131 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:20,576 : INFO : EPOCH 1 - PROGRESS: at 87.08% examples, 920073 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:21,580 : INFO : EPOCH 1 - PROGRESS: at 95.30% examples, 929252 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:22,257 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-06-14 16:17:22,269 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-06-14 16:17:22,279 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-06-14 16:17:22,283 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-06-14 16:17:22,284 : INFO : EPOCH - 1 : training on 17798082 raw words (12749777 effective words) took 13.8s, 925879 effective words/s
    2018-06-14 16:17:23,304 : INFO : EPOCH 2 - PROGRESS: at 8.00% examples, 1007006 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:24,312 : INFO : EPOCH 2 - PROGRESS: at 16.61% examples, 1043247 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:25,319 : INFO : EPOCH 2 - PROGRESS: at 24.73% examples, 1036444 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:26,323 : INFO : EPOCH 2 - PROGRESS: at 31.73% examples, 997990 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:27,344 : INFO : EPOCH 2 - PROGRESS: at 38.79% examples, 974442 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:28,345 : INFO : EPOCH 2 - PROGRESS: at 45.83% examples, 962075 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:29,345 : INFO : EPOCH 2 - PROGRESS: at 52.83% examples, 952391 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:30,351 : INFO : EPOCH 2 - PROGRESS: at 59.61% examples, 942736 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:31,357 : INFO : EPOCH 2 - PROGRESS: at 66.74% examples, 938270 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:32,357 : INFO : EPOCH 2 - PROGRESS: at 73.62% examples, 932470 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:33,366 : INFO : EPOCH 2 - PROGRESS: at 80.53% examples, 926935 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:34,378 : INFO : EPOCH 2 - PROGRESS: at 87.08% examples, 918591 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:35,390 : INFO : EPOCH 2 - PROGRESS: at 94.27% examples, 917532 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:36,174 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-06-14 16:17:36,183 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-06-14 16:17:36,192 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-06-14 16:17:36,192 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-06-14 16:17:36,193 : INFO : EPOCH - 2 : training on 17798082 raw words (12750854 effective words) took 13.9s, 917479 effective words/s
    2018-06-14 16:17:37,202 : INFO : EPOCH 3 - PROGRESS: at 7.77% examples, 984985 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:38,212 : INFO : EPOCH 3 - PROGRESS: at 16.44% examples, 1034693 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:39,216 : INFO : EPOCH 3 - PROGRESS: at 25.10% examples, 1055381 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:40,227 : INFO : EPOCH 3 - PROGRESS: at 33.83% examples, 1063504 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:41,229 : INFO : EPOCH 3 - PROGRESS: at 42.30% examples, 1067870 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:42,236 : INFO : EPOCH 3 - PROGRESS: at 50.79% examples, 1069786 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:43,243 : INFO : EPOCH 3 - PROGRESS: at 59.22% examples, 1071100 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:44,251 : INFO : EPOCH 3 - PROGRESS: at 67.75% examples, 1071922 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:45,253 : INFO : EPOCH 3 - PROGRESS: at 76.27% examples, 1073438 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:46,257 : INFO : EPOCH 3 - PROGRESS: at 84.78% examples, 1074300 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:47,261 : INFO : EPOCH 3 - PROGRESS: at 93.12% examples, 1073178 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:48,053 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-06-14 16:17:48,059 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-06-14 16:17:48,062 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-06-14 16:17:48,071 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-06-14 16:17:48,072 : INFO : EPOCH - 3 : training on 17798082 raw words (12748789 effective words) took 11.9s, 1073879 effective words/s
    2018-06-14 16:17:49,091 : INFO : EPOCH 4 - PROGRESS: at 8.00% examples, 1005623 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:50,106 : INFO : EPOCH 4 - PROGRESS: at 16.61% examples, 1038972 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:51,116 : INFO : EPOCH 4 - PROGRESS: at 25.16% examples, 1051314 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:52,122 : INFO : EPOCH 4 - PROGRESS: at 33.77% examples, 1058149 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:53,133 : INFO : EPOCH 4 - PROGRESS: at 42.24% examples, 1061681 words/s, in_qsize 6, out_qsize 1
    2018-06-14 16:17:54,141 : INFO : EPOCH 4 - PROGRESS: at 50.74% examples, 1064474 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:55,151 : INFO : EPOCH 4 - PROGRESS: at 59.17% examples, 1066074 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:56,161 : INFO : EPOCH 4 - PROGRESS: at 67.69% examples, 1067360 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:57,163 : INFO : EPOCH 4 - PROGRESS: at 75.94% examples, 1065426 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:58,166 : INFO : EPOCH 4 - PROGRESS: at 82.15% examples, 1038088 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:17:59,183 : INFO : EPOCH 4 - PROGRESS: at 88.92% examples, 1020976 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:00,189 : INFO : EPOCH 4 - PROGRESS: at 95.65% examples, 1006423 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:00,821 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-06-14 16:18:00,829 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-06-14 16:18:00,830 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-06-14 16:18:00,833 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-06-14 16:18:00,833 : INFO : EPOCH - 4 : training on 17798082 raw words (12749357 effective words) took 12.8s, 999843 effective words/s
    2018-06-14 16:18:01,847 : INFO : EPOCH 5 - PROGRESS: at 8.11% examples, 1027303 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:02,858 : INFO : EPOCH 5 - PROGRESS: at 17.00% examples, 1069428 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:03,860 : INFO : EPOCH 5 - PROGRESS: at 25.86% examples, 1086074 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:04,863 : INFO : EPOCH 5 - PROGRESS: at 34.70% examples, 1093967 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:05,865 : INFO : EPOCH 5 - PROGRESS: at 43.30% examples, 1095208 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:06,868 : INFO : EPOCH 5 - PROGRESS: at 52.04% examples, 1097950 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:07,869 : INFO : EPOCH 5 - PROGRESS: at 60.69% examples, 1100176 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:08,878 : INFO : EPOCH 5 - PROGRESS: at 68.09% examples, 1079587 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:09,888 : INFO : EPOCH 5 - PROGRESS: at 74.91% examples, 1055508 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:10,900 : INFO : EPOCH 5 - PROGRESS: at 81.64% examples, 1034528 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:11,905 : INFO : EPOCH 5 - PROGRESS: at 88.40% examples, 1018906 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:12,916 : INFO : EPOCH 5 - PROGRESS: at 95.11% examples, 1004071 words/s, in_qsize 7, out_qsize 0
    2018-06-14 16:18:13,626 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-06-14 16:18:13,635 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-06-14 16:18:13,642 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-06-14 16:18:13,643 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-06-14 16:18:13,643 : INFO : EPOCH - 5 : training on 17798082 raw words (12749291 effective words) took 12.8s, 996145 effective words/s
    2018-06-14 16:18:13,644 : INFO : training on a 88990410 raw words (63748068 effective words) took 65.1s, 978532 effective words/s
    2018-06-14 16:18:13,645 : INFO : precomputing L2-norms of word weight vectors
    2018-06-14 16:18:13,823 : INFO : saving Word2Vec object under ./data/300features_40minwords_10context, separately None
    2018-06-14 16:18:13,824 : INFO : not storing attribute vectors_norm
    2018-06-14 16:18:13,826 : INFO : not storing attribute cum_table
    2018-06-14 16:18:14,267 : INFO : saved ./data/300features_40minwords_10context


##### 4、2 预览模型

(1) 找出不匹配的词语


```python
model.doesnt_match("man woman child kitchen".split())
```
    'kitchen'




```python
model.doesnt_match("france england germany berlin".split())
```

    'berlin'




```python
model.doesnt_match("paris berlin london austria".split())
```
    'paris'



(2)匹配相似的词


```python
model.most_similar("man")
```

    [('woman', 0.6306251883506775),
     ('lady', 0.5747369527816772),
     ('lad', 0.5607120990753174),
     ('farmer', 0.5333519577980042),
     ('soldier', 0.5211611986160278),
     ('businessman', 0.5164321660995483),
     ('men', 0.514143705368042),
     ('guy', 0.5132095813751221),
     ('gentleman', 0.5126838088035583),
     ('chap', 0.5047199130058289)]

```python
model.most_similar("queen")
```
    [('princess', 0.665077805519104),
     ('belle', 0.6354636549949646),
     ('bride', 0.6334044933319092),
     ('stepmother', 0.6092513799667358),
     ('victoria', 0.603207528591156),
     ('maid', 0.5809364318847656),
     ('angela', 0.5791389346122742),
     ('showgirl', 0.5766100883483887),
     ('seductress', 0.576302170753479),
     ('maria', 0.5733033418655396)]




```python
model.most_similar("awful")
```
    [('terrible', 0.7602646350860596),
     ('atrocious', 0.7441849708557129),
     ('horrible', 0.7187424898147583),
     ('abysmal', 0.7162792682647705),
     ('dreadful', 0.6996028423309326),
     ('appalling', 0.6916007995605469),
     ('horrendous', 0.6912209987640381),
     ('horrid', 0.6631687879562378),
     ('lousy', 0.6283937692642212),
     ('amateurish', 0.616948127746582)]



（3）查看模型内部词向量内容


```python
model['flower']
```




    array([-0.04672493,  0.08882798, -0.00798039,  0.01940776,  0.00703776,
           -0.0983417 , -0.03383616,  0.01253733,  0.1013744 , -0.06382444,
            0.05575086, -0.03603694, -0.1734672 , -0.10947284,  0.04083061,
           -0.02300067, -0.08912303, -0.01208846, -0.03741381, -0.03776918,
            0.00958646, -0.04842193,  0.03746754,  0.00568479, -0.03371157,
            0.05098233,  0.03344997, -0.04607128, -0.03911143, -0.09323646,
            0.05823594,  0.05683371, -0.03013518,  0.03097427,  0.02941879,
            0.05783584,  0.02902247,  0.02647943, -0.04254698, -0.01946488,
           -0.07041467,  0.07577515,  0.01416771,  0.06094591,  0.05511365,
            0.02002226, -0.09548184,  0.01083773,  0.11513695, -0.00510942,
            0.10982258,  0.0311219 , -0.11257961, -0.04740267,  0.02418482,
           -0.13684008, -0.01806154, -0.03081693, -0.02723986, -0.02626858,
           -0.02271062,  0.02877414, -0.18940935,  0.15113354, -0.08325452,
            0.13035862,  0.03562083,  0.05715419,  0.05659889,  0.11132292,
           -0.11618023, -0.02525528, -0.04214492, -0.01676722, -0.11387489,
            0.01289947,  0.06157506,  0.02216116,  0.03087287,  0.00053991,
            0.00213429,  0.07381905, -0.09397278,  0.08932504, -0.00162343,
           -0.07650761,  0.04067037, -0.14623344, -0.01521316,  0.03096902,
           -0.00218932, -0.08328622, -0.05705699,  0.05337256,  0.03251749,
            0.06499627, -0.03883306,  0.01217427,  0.0509025 , -0.03585384,
            0.0689431 ,  0.02351258,  0.04486413, -0.01371281,  0.0296316 ,
            0.02136545, -0.14522278,  0.00962873, -0.01190588, -0.02343949,
           -0.07445157, -0.00743668,  0.06643493,  0.03143901,  0.03538921,
            0.05261549,  0.00622071,  0.00512619,  0.13138519, -0.016271  ,
            0.1280664 , -0.0281569 ,  0.05304267, -0.08197889,  0.03647416,
           -0.0908702 ,  0.0907077 ,  0.0009073 , -0.00584355,  0.0216803 ,
            0.09708863, -0.03843877,  0.07444821, -0.02166734,  0.11680321,
            0.01058747, -0.03528195, -0.01671917,  0.00431578,  0.02983457,
            0.05324177, -0.01105398, -0.0105139 ,  0.02842969, -0.02374061,
           -0.09566777,  0.00259409, -0.00075548,  0.03930911,  0.00377064,
            0.01211741, -0.04258214, -0.01305003, -0.01610391, -0.07661422,
           -0.01903511, -0.06883642, -0.08518191, -0.01282962,  0.0184722 ,
            0.04440661, -0.00693423, -0.00422318,  0.01570578, -0.00632003,
           -0.00961587, -0.02984856,  0.0214809 ,  0.07136744,  0.03213349,
           -0.04010129, -0.00035848, -0.01424925,  0.06443602,  0.02018777,
            0.01943419, -0.00966864,  0.00596202, -0.03069373, -0.01564495,
           -0.05931353,  0.02222474,  0.01904519, -0.02842978, -0.00069007,
           -0.03887554, -0.05666535,  0.01283519, -0.00330482, -0.01628766,
            0.00492181, -0.05608921, -0.01671099,  0.02535864, -0.00218894,
            0.04458237,  0.07417594,  0.03866369, -0.09731697,  0.01891821,
           -0.02167555,  0.04933357, -0.05242139,  0.03721772,  0.0229468 ,
           -0.08311929, -0.00661719, -0.00621934, -0.07366215,  0.00854658,
            0.05721372, -0.04334996,  0.00494872, -0.07360718, -0.04063034,
           -0.01522357,  0.00445656, -0.09802664, -0.05989946, -0.05275297,
           -0.03157848, -0.03841926,  0.06702825, -0.08821026,  0.06974641,
            0.05465396,  0.06257062, -0.03873832,  0.11081573, -0.05601896,
           -0.03000432, -0.02636354,  0.08552916, -0.09668133,  0.02696207,
            0.00823885, -0.02628613,  0.04240276, -0.04750042,  0.02168193,
            0.07290898, -0.02299476,  0.05413065,  0.08540422, -0.00282635,
           -0.0536682 ,  0.0743058 ,  0.00220649, -0.01906369, -0.00633779,
           -0.03175312, -0.02021488, -0.03198906, -0.04739904,  0.04052622,
            0.0988438 ,  0.00562239,  0.03735855,  0.0457716 , -0.0033249 ,
            0.01441622,  0.03325477,  0.09631445,  0.04915336,  0.05992904,
           -0.00600277, -0.01433129,  0.04592302,  0.04359821, -0.13954614,
           -0.00307374, -0.06090346, -0.13478954, -0.05491044, -0.0156491 ,
            0.13602623,  0.04711536, -0.00213679, -0.11509449, -0.11057125,
            0.02938064, -0.08248176,  0.01261425,  0.0245921 ,  0.07879733,
            0.03867466, -0.07527684, -0.03336613,  0.10514716, -0.0246095 ,
           -0.06571205,  0.05439528,  0.0557172 ,  0.02308842, -0.02448075,
            0.00975912,  0.06725417,  0.04760227, -0.06176807, -0.0582995 ],
          dtype=float32)



#### 5、使用Word2vec特征

##### 5、1  构造向量化方法


```python
import numpy as np
# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    # 对于一条评论中的每一个单词
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
# 将评论转化为向量模型
def reviews_to_vec(review_list,model):
    review_vecs=[]
    # 对于每一条评论，将其向量化
    for line in review_list:
        # 将字符串按照空格分割成列表
        #wordList=line.split()
        # 得到一条评论的矩阵向量,一条语句即可得到一个二维矩阵，行数为词的个数，列数为模型设定的维度；
        vecs = getWordVecs(line, model)
        # 根据得到的矩阵计算矩阵均值作为当前语句的特征词向量
        if len(vecs) > 0:
            vecsArray = sum(np.array(vecs)) / len(vecs)  # mean
            review_vecs.append(vecsArray)
    # 返回词向量表示的所有评论
    return review_vecs
```

##### 5、2 将数据向量化


```python
num_reviews=len(train)
clean_train_reviews=[] # 存放的是评论列表，一条评论是由单词列表组成的
for i in range(0,num_reviews):
    temp_review=review_to_wordlist(train["review"][i],True)
    clean_train_reviews.append(temp_review)
# 查看清洗后的数据
print(clean_train_reviews[0])

# 对测试集做相同的操作
num_reviews=len(test)
clean_test_reviews=[]
for i in range(0,num_reviews):
    temp_review=review_to_wordlist(test['review'][i],True)
    clean_test_reviews.append(temp_review)
```

        ['stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 'watched', 'wiz', 'watched', 'moonwalker', 'maybe', 'want', 'get', 'certain', 'insight', 'guy', 'thought', 'really', 'cool', 'eighties', 'maybe', 'make', 'mind', 'whether', 'guilty', 'innocent', 'moonwalker', 'part', 'biography', 'part', 'feature', 'film', 'remember', 'going', 'see', 'cinema', 'originally', 'released', 'subtle', 'messages', 'mj', 'feeling', 'towards', 'press', 'also', 'obvious', 'message', 'drugs', 'bad', 'kay', 'visually', 'impressive', 'course', 'michael', 'jackson', 'unless', 'remotely', 'like', 'mj', 'anyway', 'going', 'hate', 'find', 'boring', 'may', 'call', 'mj', 'egotist', 'consenting', 'making', 'movie', 'mj', 'fans', 'would', 'say', 'made', 'fans', 'true', 'really', 'nice', 'actual', 'feature', 'film', 'bit', 'finally', 'starts', 'minutes', 'excluding', 'smooth', 'criminal', 'sequence', 'joe', 'pesci', 'convincing', 'psychopathic', 'powerful', 'drug', 'lord', 'wants', 'mj', 'dead', 'bad', 'beyond', 'mj', 'overheard', 'plans', 'nah', 'joe', 'pesci', 'character', 'ranted', 'wanted', 'people', 'know', 'supplying', 'drugs', 'etc', 'dunno', 'maybe', 'hates', 'mj', 'music', 'lots', 'cool', 'things', 'like', 'mj', 'turning', 'car', 'robot', 'whole', 'speed', 'demon', 'sequence', 'also', 'director', 'must', 'patience', 'saint', 'came', 'filming', 'kiddy', 'bad', 'sequence', 'usually', 'directors', 'hate', 'working', 'one', 'kid', 'let', 'alone', 'whole', 'bunch', 'performing', 'complex', 'dance', 'scene', 'bottom', 'line', 'movie', 'people', 'like', 'mj', 'one', 'level', 'another', 'think', 'people', 'stay', 'away', 'try', 'give', 'wholesome', 'message', 'ironically', 'mj', 'bestest', 'buddy', 'movie', 'girl', 'michael', 'jackson', 'truly', 'one', 'talented', 'people', 'ever', 'grace', 'planet', 'guilty', 'well', 'attention', 'gave', 'subject', 'hmmm', 'well', 'know', 'people', 'different', 'behind', 'closed', 'doors', 'know', 'fact', 'either', 'extremely', 'nice', 'stupid', 'guy', 'one', 'sickest', 'liars', 'hope', 'latter']



```python
# 将清洗后的数据用向量表示
train_vec=reviews_to_vec(clean_train_reviews,model)
test_vec=reviews_to_vec(clean_test_reviews,model)
```

```python
print(train_vec[0])
```

    [-4.12214540e-02 -3.79349139e-02  4.92058553e-03  3.83373492e-02
     -2.45657221e-03  8.18741966e-03 -8.79610906e-03 -1.77715521e-03
      1.32088760e-02  1.10860562e-02  1.07725587e-02 -2.03826056e-03
     -6.47818862e-03  2.19975994e-03  3.60669386e-03 -6.05855137e-03
     -1.36809291e-02  4.23901244e-03 -8.60632074e-03 -1.22470358e-02
      1.90984706e-02 -1.82547256e-02 -1.34975893e-02 -6.18288175e-03
     -1.35384069e-03  1.21405336e-02  1.44483088e-02  1.13259193e-02
      1.89741399e-03  5.12676010e-03 -1.39281715e-03  3.23074499e-03
     -3.93561738e-03  8.09281687e-03  1.29367411e-02 -4.67313011e-03
      1.23364448e-02  2.59944830e-04  1.00361077e-02  3.50261860e-03
      1.37961732e-03 -5.03643784e-04 -8.41444710e-03 -1.40226181e-02
     -1.58165179e-02 -3.45300662e-03  7.24950322e-03 -1.22461706e-02
     -3.30349544e-03  7.58285267e-04  4.28325363e-03  1.03239213e-03
      1.41914198e-02 -3.84485975e-03 -3.39659879e-03 -2.38006394e-03
     -1.53069039e-04  3.38289491e-03 -9.98435193e-03 -2.60351885e-02
      1.44659057e-02 -2.39534181e-03 -7.16520778e-03  9.46791547e-03
     -1.04171550e-02 -3.98039501e-03  9.16166107e-03  3.70796656e-03
     -5.85292180e-03 -8.93773824e-03  1.18012830e-02  1.22757631e-02
     -9.76842746e-03 -1.13501498e-02 -6.58628962e-03  1.46483148e-02
      8.67670502e-03  1.49435605e-02 -1.19698187e-02 -2.33288583e-03
      4.06095162e-04 -1.06101402e-02 -1.60301849e-02  4.81574295e-03
     -8.17539989e-03 -1.66170598e-02 -9.30875331e-04  7.81632784e-03
     -9.27952358e-03 -9.36121523e-03 -1.26087192e-02 -1.80305672e-02
      1.86318907e-03 -1.93715744e-02 -1.10955165e-02 -2.71567362e-03
     -9.56838540e-03 -8.66433000e-03 -4.35272423e-03 -1.10486866e-02
     -1.72972109e-04  2.89158798e-03  3.60159835e-03  6.59811498e-03
     -6.54146849e-03 -7.31639369e-03  7.51837829e-04 -8.65462705e-03
      9.15930662e-03  8.95285620e-03  1.02896624e-02  9.78818544e-04
      5.63207011e-03  5.57235499e-03 -1.59190054e-03 -9.35532423e-03
     -3.96249171e-03  2.50648517e-02 -5.84957375e-03  7.37975151e-03
     -1.05417412e-02  1.02484963e-04 -1.50257244e-02  1.39091650e-02
     -1.17712126e-02  7.77766726e-03 -2.24176808e-03  7.05043707e-04
      8.67431739e-03  3.75606772e-03  9.08246107e-03  2.74517435e-03
     -2.35922815e-03  1.31304846e-02 -9.95927341e-03 -2.59129936e-02
      3.28526854e-03  1.28644373e-02 -1.20760067e-02  5.32581246e-03
      2.49877068e-02  4.61110624e-05 -9.94929249e-05 -5.19745509e-03
      3.84600100e-03  2.55921685e-02 -1.17436596e-03 -6.38690085e-03
      1.30926324e-02  9.25366330e-03  5.74752866e-03  3.67009494e-03
     -8.91258598e-03  4.00217369e-03  3.44281741e-03  9.58762693e-03
     -2.27720639e-03 -2.32857570e-02  1.38511115e-03 -8.95528520e-04
      4.61956041e-03  1.58334490e-03 -1.33953256e-02  9.33843276e-03
     -1.18030649e-02 -1.69549270e-02 -7.09789496e-03  2.20391827e-02
      4.45628540e-03  9.96538007e-03  1.65677878e-02 -7.59950671e-03
      2.65692874e-03 -8.20766386e-03 -7.06074345e-03 -2.07987758e-03
      5.14666043e-03 -2.76077065e-03 -1.95259660e-02 -4.92735231e-03
      8.77612858e-03  1.00959862e-02 -1.50249958e-02 -1.11764727e-02
      1.16610630e-02  7.64885400e-03  3.32668278e-03  3.01162971e-03
     -1.58037759e-03 -1.41781947e-03  8.98981516e-03  1.75613990e-03
      1.51191862e-02 -2.83710480e-03  1.07771752e-02  4.62977255e-03
     -2.14315927e-02  8.39924651e-03 -5.06151012e-03 -7.09692200e-03
     -4.08430280e-03  6.98304718e-03 -1.66823580e-02  1.03978790e-02
     -3.80447526e-03 -8.81291288e-04 -7.01620637e-03 -3.37387766e-04
      1.45387862e-03  2.47556191e-03 -5.43786858e-03  4.35696112e-03
      9.28427696e-03 -2.98430857e-03 -1.33052969e-02 -5.22909427e-03
      3.27453974e-03 -1.52650037e-02 -6.31197961e-03 -1.12691563e-02
      2.51482293e-02 -9.90015215e-03 -6.63267545e-03  1.19510287e-02
     -5.39413439e-03 -1.27368125e-04  1.43712242e-02 -6.93140865e-03
      5.03178495e-03 -6.58613742e-03  1.14176539e-03  5.67263818e-03
      5.86148271e-04  3.90761369e-03  1.96726647e-03 -5.55310410e-03
     -6.06663451e-03 -8.73890181e-03 -2.43698109e-02  6.17038008e-03
     -3.46167710e-03 -6.02825684e-03 -3.83132858e-03 -1.14010213e-02
      1.05697507e-03 -1.25030496e-02  1.31673251e-02 -1.55891930e-03
     -1.40483952e-03 -1.73462121e-03 -9.77388603e-03 -4.26941340e-04
     -4.54794628e-03 -1.76137344e-02  7.31736702e-04  4.76255040e-03
     -1.77207668e-03  7.87317303e-03  3.73680103e-03  4.42208364e-03
     -6.03005272e-03  9.67661470e-03  1.30678449e-02  4.52407294e-03
      3.43513604e-04  1.57824610e-02 -9.26215438e-04 -1.62585090e-03
      6.94119523e-04 -6.83462853e-03 -5.10692202e-03 -3.03575637e-03
      1.38966284e-03  1.88861779e-02  4.08470203e-03 -2.45477906e-02
     -2.42524100e-02  9.13219373e-03 -1.64326158e-02 -7.67338944e-03
     -1.64148907e-02  1.63434325e-02  5.82653699e-03 -8.66393099e-03
     -1.04383050e-02 -2.39454473e-02  1.32276766e-02 -1.50543327e-02
      4.05624122e-03  2.18779668e-02 -3.30929946e-04  8.58893936e-03
      8.00503015e-03  7.86819680e-04 -6.07566445e-03  3.53654486e-03
     -1.95318875e-02  1.86700427e-02  8.84436153e-03  7.88514732e-03]


#### 6、建模和预测

##### 6、1 采用随机森林模型


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)
print("Fitting a random forest to labeled training data...")
forest = forest.fit( train_vec, train['sentiment'] )
print("随机森林分类器10折交叉验证得分: ", np.mean(cross_val_score(forest, train_vec, train['sentiment'], cv=10, scoring='roc_auc')))

# 测试集
result = forest.predict(test_vec)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "./data/rf_word2vec.csv", index=False, quoting=3 )
```
    随机森林分类器10折交叉验证得分:  0.91351312


##### 6、2  采用逻辑回归模型


```python
from sklearn.linear_model import LogisticRegression as LR
model = LR()                             # 逻辑回归模型
model.fit(train_vec,train["sentiment"])
print("10折交叉验证：")
print(np.mean(cross_val_score(model,train_vec,train["sentiment"],scoring="roc_auc")))
test_predicted=np.array(model.predict(test_vec))
output=pd.DataFrame(data={"id":test["id"],"sentiment":test_predicted})
# 写入文件
output.to_csv("./data/LR_word2vec.csv",index=False,quoting=3)
```
    10折交叉验证：
    0.927272555469929

