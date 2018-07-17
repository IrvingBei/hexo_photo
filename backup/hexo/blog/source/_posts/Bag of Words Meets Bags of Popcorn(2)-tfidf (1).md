---
title: Bag of Words Meets Bags of Popcorn(2)-tfidf
date: 2018-06-14 14:33:55
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/bag.png
tags: 
      - Kaggle
      - 文本分类
categories: [Kaggle,nlp]
---
本篇是kaggle之电影评论文本情感分类（Bag of Words Meets Bags of Popcorn）实现的第二篇，语言模型选择的是TFIDF
主要参考：https://www.kaggle.com/rajathmc/bag-of-words-meets-bags-of-popcorn
https://www.cnblogs.com/lijingpeng/p/5787549.html
这两篇文章，部分地方有修改。
<!--more-->

#### 0、主要思路
（1）首先使用第一篇中数据清洗方法对训练集和测试集的数据进行清洗；前一篇http://bei.dreamcykj.com/2018/06/13/Bag%20of%20Words%20Meets%20Bags%20of%20Popcorn(1)-Bag%20of%20Words/#more

（2）利用sklearn中文本特征提取方法将数据向量化，并对训练集和测试集提取tfidf特征；

（3）选择适当的分类算法，在训练集上训练，并在测试集上进行测试，保存预测结果。


```python
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
```

#### 1、读取数据


```python
train=pd.read_csv("./data/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test=pd.read_csv("./data/testData.tsv",header=0,delimiter="\t",quoting=3)
```

#### 2、数据清洗

这里直接给出数据清洗方法，分析过程见http://bei.dreamcykj.com/2018/06/13/Bag%20of%20Words%20Meets%20Bags%20of%20Popcorn(1)-Bag%20of%20Words/#more


```python
def review_to_words(raw_review):
    # 去掉html标签
    review_text=BeautifulSoup(raw_review).get_text()
    # 去掉标点符号和非法字符
    review_text=re.sub("[^a-zA-Z]"," ",review_text)
    # 将字符全部转化为小写，并通过空格符进行分词处理
    words=review_text.lower().split()
    # 去停用词
    stops=set(stopwords.words("english"))
    meaningful_words=[w for w in words if w not in stops]
    # 将剩下的词还原成str类型
    cleaned_review=" ".join(meaningful_words)
    return cleaned_review
```

#### 3、提取特征

##### 3、1 准备数据
这一步和前一篇的类似，分别清洗训练集和测试集。


```python
# 获取数据的数量
num_reviews=len(train)
# 对数据进行清洗
clean_train_reviews=[]
print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0,num_reviews):
    if(i+1)%1000==0:
        print("Review %d of %d \n" % (i+1, num_reviews))
    temp_review=review_to_words(train["review"][i])
    clean_train_reviews.append(temp_review)

# 对测试集数据做相同的处理
num_reviews=len(test)
clean_test_reviews=[]
print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0,num_reviews):
    if(i+1)%1000==0:
        print("Review %d of %d \n"%(i+1,num_reviews))
    temp_review=review_to_words(test["review"][i])
    clean_test_reviews.append(temp_review)
print(len(clean_train_reviews))
print(len(clean_test_reviews))
```

    Cleaning and parsing the training set movie reviews...
    
    /home/xiongzy/anaconda3/lib/python3.6/site-packages/bs4/__init__.py:181: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently. 
    The code that caused this warning is on line 193 of the file /home/xiongzy/anaconda3/lib/python3.6/runpy.py. To get rid of this warning, change code that looks like this:
    
     BeautifulSoup(YOUR_MARKUP})   
    to this:
    
     BeautifulSoup(YOUR_MARKUP, "lxml")
    
      markup_type=markup_type))
    Review 1000 of 25000 
    
    Review 2000 of 25000 
    
    Review 3000 of 25000 
    
    Review 4000 of 25000 
    
    Review 5000 of 25000 
    
    Review 6000 of 25000 
    
    Review 7000 of 25000 
    
    Review 8000 of 25000 
    
    Review 9000 of 25000 
    
    Review 10000 of 25000 
    
    Review 11000 of 25000 
    
    Review 12000 of 25000 
    
    Review 13000 of 25000 
    
    Review 14000 of 25000 
    
    Review 15000 of 25000 
    
    Review 16000 of 25000 
    
    Review 17000 of 25000 
    
    Review 18000 of 25000 
    
    Review 19000 of 25000 
    
    Review 20000 of 25000 
    
    Review 21000 of 25000 
    
    Review 22000 of 25000 
    
    Review 23000 of 25000 
    
    Review 24000 of 25000 
    
    Review 25000 of 25000 
    
    Cleaning and parsing the training set movie reviews...
    
    Review 1000 of 25000 
    
    Review 2000 of 25000 
    
    Review 3000 of 25000 
    
    Review 4000 of 25000 
    
    Review 5000 of 25000 
    
    Review 6000 of 25000 
    
    Review 7000 of 25000 
    
    Review 8000 of 25000 
    
    Review 9000 of 25000 
    
    Review 10000 of 25000 
    
    Review 11000 of 25000 
    
    Review 12000 of 25000 
    
    Review 13000 of 25000 
    
    Review 14000 of 25000 
    
    Review 15000 of 25000 
    
    Review 16000 of 25000 
    
    Review 17000 of 25000 
    
    Review 18000 of 25000 
    
    Review 19000 of 25000 
    
    Review 20000 of 25000 
    
    Review 21000 of 25000 
    
    Review 22000 of 25000 
    
    Review 23000 of 25000 
    
    Review 24000 of 25000 
    
    Review 25000 of 25000 
    
    25000
    25000


##### 3、2  数据向量化


```python
from sklearn.feature_extraction.text import TfidfVectorizer as TF

tfidf=TF(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

# 合并训练和测试集以便进行tfidf向量化操作
data_all=clean_train_reviews+clean_test_reviews

# 数据向量化
print("Creating the tfidf vector...\n")
tfidf.fit(data_all)
# 获取训练集的向量表示
train_x=tfidf.transform(clean_train_reviews)
train_x=train_x.toarray()
# 获取测试集的向量表示
test_x=tfidf.transform(clean_test_reviews)
test_x=test_x.toarray()
print(train_x.shape)
print(test_x.shape)
print("finished")
```

    Creating the tfidf vector...
    
    (25000, 5000)
    (25000, 5000)
    finished


#### 4、 建模和训练
这里分别对朴素贝叶斯和逻辑回归进行了测试，发现逻辑回归的效果较好。


```python
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
```

    /home/xiongzy/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /home/xiongzy/anaconda3/lib/python3.6/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
# model=MNB(alpha=1.0, class_prior=None, fit_prior=True)     # 朴素贝叶斯模型
model = LR()                             # 逻辑回归模型
model.fit(train_x,train["sentiment"])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
from sklearn.cross_validation import cross_val_score
```


```python
import numpy as np
print("10折交叉验证：")
print(np.mean(cross_val_score(model,train_x,train["sentiment"],scoring="roc_auc")))
```

    10折交叉验证：
    0.9498602042713286



```python
test_predicted=np.array(model.predict(test_x))
```


```python
output=pd.DataFrame(data={"id":test["id"],"sentiment":test_predicted})
# 写入文件
output.to_csv("./data/tfidf.csv",index=False,quoting=3)
```
