
---
title: Bag of Words Meets Bags of Popcorn(1)-Bag of Words
date: 2018-6-13 21:26:19
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/bag.png
tags: 
      - Kaggle
      - 文本分类
categories: [Kaggle,nlp]
---
本篇是kaggle之电影评论文本情感分类（Bag of Words Meets Bags of Popcorn）的实现，主要参照Rajath Chidananda的《Bag of Words Meets Bags of Popcorn》，整体是按照他的流程来走的，对每一步都加上了注释，也对相应点给出了参考资料链接。<!--more-->

## 1、准备工作

### 1.1 查看数据格式


```python
import pandas as pd
# 读取数据
train=pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# 查看数据的大小
train.shape
```




    (25000, 3)




```python
# 查看数据的字段
train.columns
```




    Index(['id', 'sentiment', 'review'], dtype='object')




```python
# 查看数据格式
# train.head()
```


```python
# 查看完整的一条评论
train['review'][0]
```




    '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci\'s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ\'s music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ\'s bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i\'ve gave this subject....hmmm well i don\'t know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."'



### 1.2 数据清洗

观察这条数据，发现有html标签，需要从html标签文件中抽取出文本，有标点符号，有大小写，介词、冠词较多，那么接下来就对数据进行清洗，

#### 1.2.1 清洗1——去除html标签


```python
from bs4 import BeautifulSoup
example1=BeautifulSoup(train['review'][0])
```

    /home/xiongzy/anaconda3/lib/python3.6/site-packages/bs4/__init__.py:181: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    The code that caused this warning is on line 193 of the file /home/xiongzy/anaconda3/lib/python3.6/runpy.py. To get rid of this warning, change code that looks like this:
    
     BeautifulSoup(YOUR_MARKUP})
    
    to this:
    
     BeautifulSoup(YOUR_MARKUP, "lxml")
    
      markup_type=markup_type))


example1就是bs对象，调用get_text()方法即可获取文本
我们首先再输出一下原始文本，然后输出第一次清洗后的文本


```python
train['review'][0]
```




    '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci\'s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ\'s music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ\'s bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i\'ve gave this subject....hmmm well i don\'t know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."'




```python
example1.get_text()
```




    '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci\'s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ\'s music.Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ\'s bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i\'ve gave this subject....hmmm well i don\'t know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."'



#### 1.2.2 清洗2——去掉标点符号和非法字符

然后发现文本中还有一些标点符号啥的，可以考虑用正则表达式去掉

re.sub(pattern, repl, string, count=0, flags=0)

pattern：表示正则表达式中的模式字符串；

repl：被替换的字符串（既可以是字符串，也可以是函数）；

string：要被处理的，要被替换的字符串；

count：匹配的次数, 默认是全部替换

flags：具体用处不详

具体strip(),replace()和re.sub()用法:https://www.cnblogs.com/sshcy/p/8065113.html


```python
import re 
letters_only=re.sub("[^a-zA-Z]"," ",example1.get_text()) # 注意，这里是将非字母字符替换成空白字符
print(letters_only)
```

     With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring  Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him The actual feature film bit when it finally starts is only on for    minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord  Why he wants MJ dead so bad is beyond me  Because MJ overheard his plans  Nah  Joe Pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno  maybe he just hates MJ s music Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence  Also  the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene Bottom line  this movie is for people who like MJ on one level or another  which i think is most people   If not  then stay away  It does try and give off a wholesome message and ironically MJ s bestest buddy in this movie is a girl  Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty  Well  with all the attention i ve gave this subject    hmmm well i don t know because people can be different behind closed doors  i know this for a fact  He is either an extremely nice but stupid guy or one of the most sickest liars  I hope he is not the latter  


#### 1.2.3 清洗3——数据归一化

看到评论中数据，每条评论记录中的单词大小写都存在，于是可以考虑把所以字符改为小写，然后再以空格为分割符进行切割,得到一条评论的一个列表，列表元素为这条评论的单词


```python
lower_case=letters_only.lower()
words=lower_case.split() # split默认是以空格为分割符
# print(words)
```

#### 1.2.4 清洗4——去停用词

因为是英文数据，于是可以考虑直接使用nltk提供的英文停用词表，nltk里好像有9门语言的停用词表，目前不支持中文。

使用停用词表需要预先下载ntlk_data,我已经上传百度云:链接：https://pan.baidu.com/s/1E-eKAZg5aEBRTqSYG24Tng 密码：lrbe


```python
# 查看停用词表中的单词
import nltk
# from nltk.corpus import stopwords
# print(stopwords.words("english"))
# words = [w for w in words if not w in stopwords.words("english")]
print(words)
```

    ['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'started', 'listening', 'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary', 'here', 'and', 'there', 'watched', 'the', 'wiz', 'and', 'watched', 'moonwalker', 'again', 'maybe', 'i', 'just', 'want', 'to', 'get', 'a', 'certain', 'insight', 'into', 'this', 'guy', 'who', 'i', 'thought', 'was', 'really', 'cool', 'in', 'the', 'eighties', 'just', 'to', 'maybe', 'make', 'up', 'my', 'mind', 'whether', 'he', 'is', 'guilty', 'or', 'innocent', 'moonwalker', 'is', 'part', 'biography', 'part', 'feature', 'film', 'which', 'i', 'remember', 'going', 'to', 'see', 'at', 'the', 'cinema', 'when', 'it', 'was', 'originally', 'released', 'some', 'of', 'it', 'has', 'subtle', 'messages', 'about', 'mj', 's', 'feeling', 'towards', 'the', 'press', 'and', 'also', 'the', 'obvious', 'message', 'of', 'drugs', 'are', 'bad', 'm', 'kay', 'visually', 'impressive', 'but', 'of', 'course', 'this', 'is', 'all', 'about', 'michael', 'jackson', 'so', 'unless', 'you', 'remotely', 'like', 'mj', 'in', 'anyway', 'then', 'you', 'are', 'going', 'to', 'hate', 'this', 'and', 'find', 'it', 'boring', 'some', 'may', 'call', 'mj', 'an', 'egotist', 'for', 'consenting', 'to', 'the', 'making', 'of', 'this', 'movie', 'but', 'mj', 'and', 'most', 'of', 'his', 'fans', 'would', 'say', 'that', 'he', 'made', 'it', 'for', 'the', 'fans', 'which', 'if', 'true', 'is', 'really', 'nice', 'of', 'him', 'the', 'actual', 'feature', 'film', 'bit', 'when', 'it', 'finally', 'starts', 'is', 'only', 'on', 'for', 'minutes', 'or', 'so', 'excluding', 'the', 'smooth', 'criminal', 'sequence', 'and', 'joe', 'pesci', 'is', 'convincing', 'as', 'a', 'psychopathic', 'all', 'powerful', 'drug', 'lord', 'why', 'he', 'wants', 'mj', 'dead', 'so', 'bad', 'is', 'beyond', 'me', 'because', 'mj', 'overheard', 'his', 'plans', 'nah', 'joe', 'pesci', 's', 'character', 'ranted', 'that', 'he', 'wanted', 'people', 'to', 'know', 'it', 'is', 'he', 'who', 'is', 'supplying', 'drugs', 'etc', 'so', 'i', 'dunno', 'maybe', 'he', 'just', 'hates', 'mj', 's', 'music', 'lots', 'of', 'cool', 'things', 'in', 'this', 'like', 'mj', 'turning', 'into', 'a', 'car', 'and', 'a', 'robot', 'and', 'the', 'whole', 'speed', 'demon', 'sequence', 'also', 'the', 'director', 'must', 'have', 'had', 'the', 'patience', 'of', 'a', 'saint', 'when', 'it', 'came', 'to', 'filming', 'the', 'kiddy', 'bad', 'sequence', 'as', 'usually', 'directors', 'hate', 'working', 'with', 'one', 'kid', 'let', 'alone', 'a', 'whole', 'bunch', 'of', 'them', 'performing', 'a', 'complex', 'dance', 'scene', 'bottom', 'line', 'this', 'movie', 'is', 'for', 'people', 'who', 'like', 'mj', 'on', 'one', 'level', 'or', 'another', 'which', 'i', 'think', 'is', 'most', 'people', 'if', 'not', 'then', 'stay', 'away', 'it', 'does', 'try', 'and', 'give', 'off', 'a', 'wholesome', 'message', 'and', 'ironically', 'mj', 's', 'bestest', 'buddy', 'in', 'this', 'movie', 'is', 'a', 'girl', 'michael', 'jackson', 'is', 'truly', 'one', 'of', 'the', 'most', 'talented', 'people', 'ever', 'to', 'grace', 'this', 'planet', 'but', 'is', 'he', 'guilty', 'well', 'with', 'all', 'the', 'attention', 'i', 've', 'gave', 'this', 'subject', 'hmmm', 'well', 'i', 'don', 't', 'know', 'because', 'people', 'can', 'be', 'different', 'behind', 'closed', 'doors', 'i', 'know', 'this', 'for', 'a', 'fact', 'he', 'is', 'either', 'an', 'extremely', 'nice', 'but', 'stupid', 'guy', 'or', 'one', 'of', 'the', 'most', 'sickest', 'liars', 'i', 'hope', 'he', 'is', 'not', 'the', 'latter']


### 1.3 小结
首先对数据的格式、基本信息进行了观察，主要如下：

（1）数据大小：(25000, 3)

（2）每条记录的字段：['id', 'sentiment','review']

接着对数据进行了清洗：去除html标签、去标点符号、转化为小写、去停用词，可以把数据清理这个过程写成一个函数


```python
# 需要映入的库
import pandas as pd
from bs4 import BeautifulSoup
import re 
import nltk
from nltk.corpus import stopwords
```


```python
# 数据清理函数

def review_to_words(raw_review):
    # 通过BeautifulSoup抽取纯文本
    review_text=BeautifulSoup(raw_review).get_text()
    # 通过正则表达式去掉标点符号
    letters_only=re.sub("[^a-zA-Z]"," ",review_text)
    # 转化为小写，并进行分词
    words=letters_only.lower().split()
    # 去停用词
    stops = set(stopwords.words("english"))
    meaningful_words=[w for w in words if w not in stops]
   
    # 将处理后的数据返回
    cleaned_review=" ".join(meaningful_words)
    return cleaned_review

```


```python
# 对编写的函数进行测试
clean_review = review_to_words(train["review"][0])
print(clean_review)
```

    stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter


    /home/xiongzy/anaconda3/lib/python3.6/site-packages/bs4/__init__.py:181: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    The code that caused this warning is on line 193 of the file /home/xiongzy/anaconda3/lib/python3.6/runpy.py. To get rid of this warning, change code that looks like this:
    
     BeautifulSoup(YOUR_MARKUP})
    
    to this:
    
     BeautifulSoup(YOUR_MARKUP, "lxml")
    
      markup_type=markup_type))


## 2、特征提取
经过前面的分析，可以通过前面定义的方法来清洗数据。数据清洗好后，接下来就是提取每一条评论数据的特征，这里采用词带特征

### 2.1 准备数据
使用在准备工作中定义的方法来清洗数据


```python
# 获取数据的数量
num_reviews=len(train)
# 对数据进行清洗
clean_train_reviews=[]
print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0,num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d \n" % (i+1, num_reviews))
    temp_review=review_to_words(train["review"][i])
    clean_train_reviews.append(temp_review)
clean_train_reviews[:5]
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
    





    ['stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter',
     'classic war worlds timothy hines entertaining film obviously goes great effort lengths faithfully recreate h g wells classic book mr hines succeeds watched film appreciated fact standard predictable hollywood fare comes every year e g spielberg version tom cruise slightest resemblance book obviously everyone looks different things movie envision amateur critics look criticize everything others rate movie important bases like entertained people never agree critics enjoyed effort mr hines put faithful h g wells classic novel found entertaining made easy overlook critics perceive shortcomings',
     'film starts manager nicholas bell giving welcome investors robert carradine primal park secret project mutating primal animal using fossilized dna like jurassik park scientists resurrect one nature fearsome predators sabretooth tiger smilodon scientific ambition turns deadly however high voltage fence opened creature escape begins savagely stalking prey human visitors tourists scientific meanwhile youngsters enter restricted area security center attacked pack large pre historical animals deadlier bigger addition security agent stacy haiduk mate brian wimmer fight hardly carnivorous smilodons sabretooths course real star stars astounding terrifyingly though convincing giant animals savagely stalking prey group run afoul fight one nature fearsome predators furthermore third sabretooth dangerous slow stalks victims movie delivers goods lots blood gore beheading hair raising chills full scares sabretooths appear mediocre special effects story provides exciting stirring entertainment results quite boring giant animals majority made computer generator seem totally lousy middling performances though players reacting appropriately becoming food actors give vigorously physical performances dodging beasts running bound leaps dangling walls packs ridiculous final deadly scene small kids realistic gory violent attack scenes films sabretooths smilodon following sabretooth james r hickox vanessa angel david keith john rhys davies much better bc roland emmerich steven strait cliff curtis camilla belle motion picture filled bloody moments badly directed george miller originality takes many elements previous films miller australian director usually working television tidal wave journey center earth many others occasionally cinema man snowy river zeus roxanne robinson crusoe rating average bottom barrel',
     'must assumed praised film greatest filmed opera ever read somewhere either care opera care wagner care anything except desire appear cultured either representation wagner swan song movie strikes unmitigated disaster leaden reading score matched tricksy lugubrious realisation text questionable people ideas opera matter play especially one shakespeare allowed anywhere near theatre film studio syberberg fashionably without smallest justification wagner text decided parsifal bisexual integration title character latter stages transmutes kind beatnik babe though one continues sing high tenor actors film singers get double dose armin jordan conductor seen face heard voice amfortas also appears monstrously double exposure kind batonzilla conductor ate monsalvat playing good friday music way transcendant loveliness nature represented scattering shopworn flaccid crocuses stuck ill laid turf expedient baffles theatre sometimes piece imperfections thoughts think syberberg splice parsifal gurnemanz mountain pasture lush provided julie andrews sound music sound hard endure high voices trumpets particular possessing aural glare adds another sort fatigue impatience uninspired conducting paralytic unfolding ritual someone another review mentioned bayreuth recording knappertsbusch though tempi often slow jordan altogether lacks sense pulse feeling ebb flow music half century orchestral sound set modern pressings still superior film',
     'superbly trashy wondrously unpretentious exploitation hooray pre credits opening sequences somewhat give false impression dealing serious harrowing drama need fear barely ten minutes later necks nonsensical chainsaw battles rough fist fights lurid dialogs gratuitous nudity bo ingrid two orphaned siblings unusually close even slightly perverted relationship imagine playfully ripping towel covers sister naked body stare unshaven genitals several whole minutes well bo sister judging dubbed laughter mind sick dude anyway kids fled russia parents nasty soldiers brutally slaughtered mommy daddy friendly smuggler took custody however even raised trained bo ingrid expert smugglers actual plot lifts years later facing ultimate quest mythical incredibly valuable white fire diamond coincidentally found mine things life ever made little sense plot narrative structure white fire sure lot fun watch time clue beating cause bet actors understood even less whatever violence magnificently grotesque every single plot twist pleasingly retarded script goes totally bonkers beyond repair suddenly reveal reason bo needs replacement ingrid fred williamson enters scene big cigar mouth sleazy black fingers local prostitutes bo principal opponent italian chick big breasts hideous accent preposterous catchy theme song plays least dozen times throughout film obligatory falling love montage loads attractions god brilliant experience original french title translates life survive uniquely appropriate makes much sense rest movie none']



### 2.2 提取数据特征
这里借助于sklearn来提取数据的词带特征,Scikit Learn CountVectorizer 入门实例:
https://blog.csdn.net/guotong1988/article/details/51567562

参数介绍：
http://www.itkeyword.com/doc/4813494854317445586/TfidfVectorizer-sklearn-CountVectorizer


```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
print("Creating the bag of words...\n")
# 将每一条评论的特征矩阵转化为数组
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
# 查看数据特征维度
print(train_data_features.shape)
```

    Creating the bag of words...
    
    (25000, 5000)


也就是每条记录提取了5000个特征。

接下来可以查看特征的名字,因为我们创建的是词带特征，因此所有的特征组成词汇表

也可以对特征进行统计，统计每个特征出现的次数


```python
vocab=vectorizer.get_feature_names()
print(vocab)
```

    ['abandoned', 'abc', 'abilities', 'ability', 'able', 'abraham', 'absence', 'absent', 'absolute', 'absolutely', 'absurd', 'abuse', 'abusive', 'abysmal', 'academy', 'accent', 'accents', 'accept', 'acceptable', 'accepted', 'access', 'accident', 'accidentally', 'accompanied', 'accomplished', 'according', 'account', 'accuracy', 'accurate', 'accused', 'achieve', 'achieved', 'achievement', 'acid', 'across', 'act', 'acted', 'acting', 'action', 'actions', 'activities', 'actor', 'actors', 'actress', 'actresses', 'acts', 'actual', 'actually', 'ad', 'adam', 'adams', 'adaptation', 'adaptations', 'adapted', 'add', 'added', 'adding', 'addition', 'adds', 'adequate', 'admire', 'admit', 'admittedly', 'adorable', 'adult', 'adults', 'advance', 'advanced', 'advantage', 'adventure', 'adventures', 'advertising', 'advice', 'advise', 'affair', 'affect', 'affected', 'afford', 'aforementioned', 'afraid', 'africa', 'african', 'afternoon', 'afterwards', 'age', 'aged', 'agent', 'agents', 'ages', 'aging', 'ago', 'agree', 'agreed', 'agrees', 'ah', 'ahead', 'aid', 'aids', 'aim', 'aimed', 'air', 'aired', 'airplane', 'airport', 'aka', 'akshay', 'al', 'alan', 'alas', 'albeit', 'albert', 'album', 'alcohol', 'alcoholic', 'alec', 'alert', 'alex', 'alexander', 'alfred', 'alice', 'alicia', 'alien', 'aliens', 'alike', 'alison', 'alive', 'allen', 'allow', 'allowed', 'allowing', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'alright', 'also', 'alternate', 'alternative', 'although', 'altman', 'altogether', 'always', 'amanda', 'amateur', 'amateurish', 'amazed', 'amazing', 'amazingly', 'ambiguous', 'ambitious', 'america', 'american', 'americans', 'amitabh', 'among', 'amongst', 'amount', 'amounts', 'amused', 'amusing', 'amy', 'analysis', 'ancient', 'anderson', 'andre', 'andrew', 'andrews', 'andy', 'angel', 'angela', 'angeles', 'angels', 'anger', 'angle', 'angles', 'angry', 'animal', 'animals', 'animated', 'animation', 'anime', 'ann', 'anna', 'anne', 'annie', 'annoyed', 'annoying', 'another', 'answer', 'answers', 'anthony', 'anti', 'antics', 'antonioni', 'antwone', 'anybody', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'apartment', 'ape', 'apes', 'appalling', 'apparent', 'apparently', 'appeal', 'appealing', 'appear', 'appearance', 'appearances', 'appeared', 'appearing', 'appears', 'appreciate', 'appreciated', 'appreciation', 'approach', 'appropriate', 'april', 'area', 'areas', 'arguably', 'argue', 'argument', 'arm', 'armed', 'arms', 'army', 'arnold', 'around', 'arrested', 'arrival', 'arrive', 'arrived', 'arrives', 'arrogant', 'art', 'arthur', 'artificial', 'artist', 'artistic', 'artists', 'arts', 'ashamed', 'ashley', 'asian', 'aside', 'ask', 'asked', 'asking', 'asks', 'asleep', 'aspect', 'aspects', 'ass', 'assassin', 'assault', 'assigned', 'assistant', 'associated', 'assume', 'assumed', 'astaire', 'astonishing', 'atlantis', 'atmosphere', 'atmospheric', 'atrocious', 'attached', 'attack', 'attacked', 'attacks', 'attempt', 'attempted', 'attempting', 'attempts', 'attend', 'attention', 'attitude', 'attitudes', 'attorney', 'attracted', 'attraction', 'attractive', 'audience', 'audiences', 'audio', 'aunt', 'austen', 'austin', 'australia', 'australian', 'authentic', 'author', 'authority', 'available', 'average', 'avoid', 'avoided', 'awake', 'award', 'awards', 'aware', 'away', 'awe', 'awesome', 'awful', 'awfully', 'awhile', 'awkward', 'babe', 'baby', 'bacall', 'back', 'backdrop', 'background', 'backgrounds', 'bad', 'badly', 'bag', 'baker', 'bakshi', 'balance', 'baldwin', 'ball', 'ballet', 'balls', 'band', 'bands', 'bang', 'bank', 'banned', 'bar', 'barbara', 'bare', 'barely', 'bargain', 'barry', 'barrymore', 'base', 'baseball', 'based', 'basement', 'basic', 'basically', 'basis', 'basketball', 'bat', 'bath', 'bathroom', 'batman', 'battle', 'battles', 'bay', 'bbc', 'beach', 'bear', 'bears', 'beast', 'beat', 'beaten', 'beating', 'beats', 'beatty', 'beautiful', 'beautifully', 'beauty', 'became', 'become', 'becomes', 'becoming', 'bed', 'bedroom', 'beer', 'began', 'begin', 'beginning', 'begins', 'behave', 'behavior', 'behind', 'beings', 'bela', 'belief', 'beliefs', 'believable', 'believe', 'believed', 'believes', 'believing', 'bell', 'belong', 'belongs', 'beloved', 'belushi', 'ben', 'beneath', 'benefit', 'bergman', 'berlin', 'besides', 'best', 'bet', 'bette', 'better', 'bettie', 'betty', 'beyond', 'bible', 'big', 'bigger', 'biggest', 'biko', 'bill', 'billed', 'billy', 'bin', 'biography', 'bird', 'birds', 'birth', 'birthday', 'bit', 'bite', 'bits', 'bitter', 'bizarre', 'black', 'blade', 'blah', 'blair', 'blake', 'blame', 'bland', 'blank', 'blast', 'blatant', 'bleak', 'blend', 'blew', 'blind', 'blob', 'block', 'blockbuster', 'blond', 'blonde', 'blood', 'bloody', 'blow', 'blowing', 'blown', 'blows', 'blue', 'blues', 'blunt', 'bo', 'board', 'boat', 'bob', 'bobby', 'bodies', 'body', 'bold', 'boll', 'bollywood', 'bomb', 'bond', 'bone', 'bonus', 'book', 'books', 'boom', 'boot', 'border', 'bore', 'bored', 'boredom', 'boring', 'born', 'borrowed', 'boss', 'bother', 'bothered', 'bottle', 'bottom', 'bought', 'bound', 'bourne', 'box', 'boxing', 'boy', 'boyfriend', 'boyle', 'boys', 'brad', 'brady', 'brain', 'brains', 'branagh', 'brand', 'brando', 'brave', 'brazil', 'break', 'breaking', 'breaks', 'breasts', 'breath', 'breathtaking', 'brenda', 'brian', 'bride', 'bridge', 'brief', 'briefly', 'bright', 'brilliance', 'brilliant', 'brilliantly', 'bring', 'bringing', 'brings', 'britain', 'british', 'broad', 'broadcast', 'broadway', 'broke', 'broken', 'brooklyn', 'brooks', 'brosnan', 'brother', 'brothers', 'brought', 'brown', 'bruce', 'brutal', 'brutality', 'brutally', 'buck', 'bucks', 'bud', 'buddies', 'buddy', 'budget', 'buff', 'buffalo', 'buffs', 'bug', 'bugs', 'build', 'building', 'buildings', 'builds', 'built', 'bull', 'bullet', 'bullets', 'bumbling', 'bunch', 'bunny', 'buried', 'burn', 'burned', 'burning', 'burns', 'burt', 'burton', 'bus', 'bush', 'business', 'businessman', 'buster', 'busy', 'butler', 'butt', 'button', 'buy', 'buying', 'cabin', 'cable', 'cage', 'cagney', 'caine', 'cake', 'caliber', 'california', 'call', 'called', 'calling', 'calls', 'calm', 'came', 'cameo', 'cameos', 'camera', 'cameras', 'cameron', 'camp', 'campbell', 'campy', 'canada', 'canadian', 'candy', 'cannibal', 'cannot', 'cant', 'capable', 'capital', 'captain', 'captivating', 'capture', 'captured', 'captures', 'capturing', 'car', 'card', 'cardboard', 'cards', 'care', 'cared', 'career', 'careers', 'careful', 'carefully', 'carell', 'cares', 'caring', 'carl', 'carla', 'carol', 'carpenter', 'carradine', 'carrey', 'carrie', 'carried', 'carries', 'carry', 'carrying', 'cars', 'carter', 'cartoon', 'cartoons', 'cary', 'case', 'cases', 'cash', 'cassidy', 'cast', 'casting', 'castle', 'cat', 'catch', 'catches', 'catching', 'catchy', 'category', 'catherine', 'catholic', 'cats', 'caught', 'cause', 'caused', 'causes', 'causing', 'cave', 'cd', 'celebrity', 'cell', 'celluloid', 'center', 'centered', 'centers', 'central', 'century', 'certain', 'certainly', 'cg', 'cgi', 'chain', 'chair', 'challenge', 'challenging', 'championship', 'chan', 'chance', 'chances', 'change', 'changed', 'changes', 'changing', 'channel', 'channels', 'chaos', 'chaplin', 'chapter', 'character', 'characterization', 'characters', 'charge', 'charisma', 'charismatic', 'charles', 'charlie', 'charlotte', 'charm', 'charming', 'chase', 'chased', 'chases', 'chasing', 'che', 'cheap', 'cheated', 'cheating', 'check', 'checked', 'checking', 'cheek', 'cheese', 'cheesy', 'chemistry', 'chess', 'chest', 'chicago', 'chick', 'chicken', 'chicks', 'chief', 'child', 'childhood', 'childish', 'children', 'chilling', 'china', 'chinese', 'choice', 'choices', 'choose', 'chooses', 'choreographed', 'choreography', 'chorus', 'chose', 'chosen', 'chris', 'christ', 'christian', 'christianity', 'christians', 'christmas', 'christopher', 'christy', 'chuck', 'church', 'cia', 'cinderella', 'cinema', 'cinematic', 'cinematographer', 'cinematography', 'circle', 'circumstances', 'cities', 'citizen', 'city', 'civil', 'civilization', 'claim', 'claimed', 'claims', 'claire', 'clark', 'class', 'classes', 'classic', 'classical', 'classics', 'claus', 'clean', 'clear', 'clearly', 'clever', 'cleverly', 'clich', 'cliche', 'cliff', 'climactic', 'climax', 'clint', 'clip', 'clips', 'clock', 'close', 'closed', 'closely', 'closer', 'closest', 'closet', 'closing', 'clothes', 'clothing', 'clown', 'club', 'clue', 'clues', 'clumsy', 'co', 'coach', 'coast', 'code', 'coffee', 'coherent', 'cold', 'cole', 'collection', 'college', 'colonel', 'color', 'colorful', 'colors', 'colour', 'columbo', 'com', 'combat', 'combination', 'combine', 'combined', 'come', 'comedian', 'comedic', 'comedies', 'comedy', 'comes', 'comfort', 'comfortable', 'comic', 'comical', 'comics', 'coming', 'command', 'comment', 'commentary', 'commented', 'comments', 'commercial', 'commercials', 'commit', 'committed', 'common', 'communist', 'community', 'companies', 'companion', 'company', 'compare', 'compared', 'comparing', 'comparison', 'compassion', 'compelled', 'compelling', 'competent', 'competition', 'complain', 'complaint', 'complete', 'completely', 'complex', 'complexity', 'complicated', 'composed', 'composer', 'computer', 'con', 'conceived', 'concept', 'concern', 'concerned', 'concerning', 'concerns', 'concert', 'conclusion', 'condition', 'conditions', 'confess', 'confidence', 'conflict', 'conflicts', 'confrontation', 'confused', 'confusing', 'confusion', 'connect', 'connected', 'connection', 'connery', 'conscious', 'consequences', 'conservative', 'consider', 'considerable', 'considered', 'considering', 'consistent', 'consistently', 'consists', 'conspiracy', 'constant', 'constantly', 'constructed', 'construction', 'contact', 'contain', 'contained', 'contains', 'contemporary', 'content', 'contest', 'context', 'continue', 'continued', 'continues', 'continuity', 'contract', 'contrary', 'contrast', 'contrived', 'control', 'controversial', 'conventional', 'conversation', 'conversations', 'convey', 'convince', 'convinced', 'convincing', 'convincingly', 'convoluted', 'cook', 'cool', 'cooper', 'cop', 'copies', 'cops', 'copy', 'core', 'corner', 'corny', 'corporate', 'corpse', 'correct', 'correctly', 'corrupt', 'corruption', 'cost', 'costs', 'costume', 'costumes', 'could', 'count', 'counter', 'countless', 'countries', 'country', 'countryside', 'couple', 'couples', 'courage', 'course', 'court', 'cousin', 'cover', 'covered', 'covers', 'cowboy', 'cox', 'crack', 'cracking', 'craft', 'crafted', 'craig', 'crap', 'crappy', 'crash', 'craven', 'crawford', 'crazed', 'crazy', 'cream', 'create', 'created', 'creates', 'creating', 'creation', 'creative', 'creativity', 'creator', 'creators', 'creature', 'creatures', 'credibility', 'credible', 'credit', 'credits', 'creep', 'creepy', 'crew', 'cried', 'crime', 'crimes', 'criminal', 'criminals', 'cringe', 'crisis', 'critic', 'critical', 'criticism', 'critics', 'crocodile', 'cross', 'crowd', 'crucial', 'crude', 'cruel', 'cruise', 'crush', 'cry', 'crying', 'crystal', 'cuba', 'cube', 'cult', 'cultural', 'culture', 'cup', 'cure', 'curiosity', 'curious', 'current', 'currently', 'curse', 'curtis', 'cusack', 'cut', 'cute', 'cuts', 'cutting', 'cynical', 'da', 'dad', 'daddy', 'daily', 'dalton', 'damage', 'damme', 'damn', 'damon', 'dan', 'dana', 'dance', 'dancer', 'dancers', 'dances', 'dancing', 'danes', 'danger', 'dangerous', 'daniel', 'danny', 'dare', 'daring', 'dark', 'darker', 'darkness', 'darren', 'date', 'dated', 'dating', 'daughter', 'daughters', 'dave', 'david', 'davies', 'davis', 'dawn', 'dawson', 'day', 'days', 'de', 'dead', 'deadly', 'deaf', 'deal', 'dealing', 'deals', 'dealt', 'dean', 'dear', 'death', 'deaths', 'debut', 'decade', 'decades', 'deceased', 'decent', 'decide', 'decided', 'decides', 'decision', 'decisions', 'dedicated', 'dee', 'deep', 'deeper', 'deeply', 'defeat', 'defend', 'defense', 'defined', 'definite', 'definitely', 'definition', 'degree', 'del', 'deliberately', 'delight', 'delightful', 'deliver', 'delivered', 'delivering', 'delivers', 'delivery', 'demand', 'demands', 'demented', 'demise', 'demon', 'demons', 'deniro', 'dennis', 'dentist', 'denzel', 'department', 'depicted', 'depicting', 'depiction', 'depicts', 'depressed', 'depressing', 'depression', 'depth', 'der', 'derek', 'descent', 'describe', 'described', 'describes', 'description', 'desert', 'deserve', 'deserved', 'deserves', 'design', 'designed', 'designs', 'desire', 'desired', 'despair', 'desperate', 'desperately', 'desperation', 'despite', 'destiny', 'destroy', 'destroyed', 'destroying', 'destruction', 'detail', 'detailed', 'details', 'detective', 'determined', 'develop', 'developed', 'developing', 'development', 'develops', 'device', 'devil', 'devoid', 'devoted', 'dialog', 'dialogs', 'dialogue', 'dialogues', 'diamond', 'diana', 'diane', 'dick', 'dickens', 'die', 'died', 'dies', 'difference', 'differences', 'different', 'difficult', 'dig', 'digital', 'dignity', 'dimension', 'dimensional', 'din', 'dinner', 'dinosaur', 'dinosaurs', 'dire', 'direct', 'directed', 'directing', 'direction', 'directions', 'directly', 'director', 'directorial', 'directors', 'directs', 'dirty', 'disagree', 'disappear', 'disappeared', 'disappoint', 'disappointed', 'disappointing', 'disappointment', 'disaster', 'disbelief', 'disc', 'discover', 'discovered', 'discovers', 'discovery', 'discuss', 'discussion', 'disease', 'disgusting', 'disjointed', 'dislike', 'disliked', 'disney', 'display', 'displayed', 'displays', 'distance', 'distant', 'distinct', 'distracting', 'distribution', 'disturbed', 'disturbing', 'divorce', 'dixon', 'doc', 'doctor', 'documentaries', 'documentary', 'dog', 'dogs', 'doll', 'dollar', 'dollars', 'dolls', 'dolph', 'domestic', 'domino', 'donald', 'done', 'donna', 'doo', 'doom', 'doomed', 'door', 'doors', 'dorothy', 'double', 'doubt', 'doubts', 'douglas', 'downey', 'downhill', 'downright', 'dozen', 'dozens', 'dr', 'dracula', 'drag', 'dragged', 'dragon', 'drags', 'drake', 'drama', 'dramas', 'dramatic', 'draw', 'drawing', 'drawn', 'draws', 'dreadful', 'dream', 'dreams', 'dreary', 'dreck', 'dress', 'dressed', 'dressing', 'drew', 'drink', 'drinking', 'drive', 'drivel', 'driven', 'driver', 'drives', 'driving', 'drop', 'dropped', 'dropping', 'drops', 'drug', 'drugs', 'drunk', 'drunken', 'dry', 'dub', 'dubbed', 'dubbing', 'dud', 'dude', 'due', 'duke', 'dull', 'dumb', 'duo', 'dust', 'dutch', 'duty', 'dvd', 'dying', 'dynamic', 'eager', 'ear', 'earl', 'earlier', 'early', 'earned', 'ears', 'earth', 'ease', 'easier', 'easily', 'east', 'eastern', 'eastwood', 'easy', 'eat', 'eaten', 'eating', 'eccentric', 'ed', 'eddie', 'edgar', 'edge', 'edgy', 'edie', 'edited', 'editing', 'edition', 'editor', 'education', 'educational', 'edward', 'eerie', 'effect', 'effective', 'effectively', 'effects', 'effort', 'efforts', 'ego', 'eight', 'eighties', 'either', 'elaborate', 'elderly', 'elegant', 'element', 'elements', 'elephant', 'elizabeth', 'ellen', 'elm', 'else', 'elsewhere', 'elvira', 'elvis', 'em', 'embarrassed', 'embarrassing', 'embarrassment', 'emily', 'emma', 'emotion', 'emotional', 'emotionally', 'emotions', 'empathy', 'emperor', 'emphasis', 'empire', 'empty', 'en', 'encounter', 'encounters', 'end', 'endearing', 'ended', 'ending', 'endings', 'endless', 'ends', 'endure', 'enemies', 'enemy', 'energy', 'engage', 'engaged', 'engaging', 'england', 'english', 'enjoy', 'enjoyable', 'enjoyed', 'enjoying', 'enjoyment', 'enjoys', 'enormous', 'enough', 'ensemble', 'ensues', 'enter', 'enterprise', 'enters', 'entertain', 'entertained', 'entertaining', 'entertainment', 'enthusiasm', 'entire', 'entirely', 'entry', 'environment', 'epic', 'episode', 'episodes', 'equal', 'equally', 'equipment', 'equivalent', 'er', 'era', 'eric', 'erotic', 'errors', 'escape', 'escaped', 'escapes', 'especially', 'essence', 'essential', 'essentially', 'established', 'estate', 'et', 'etc', 'ethan', 'eugene', 'europe', 'european', 'eva', 'eve', 'even', 'evening', 'event', 'events', 'eventually', 'ever', 'every', 'everybody', 'everyday', 'everyone', 'everything', 'everywhere', 'evidence', 'evident', 'evil', 'ex', 'exact', 'exactly', 'exaggerated', 'examination', 'example', 'examples', 'excellent', 'except', 'exception', 'exceptional', 'exceptionally', 'excessive', 'exchange', 'excited', 'excitement', 'exciting', 'excuse', 'executed', 'execution', 'executive', 'exercise', 'exist', 'existed', 'existence', 'existent', 'exists', 'exotic', 'expect', 'expectations', 'expected', 'expecting', 'expedition', 'expensive', 'experience', 'experienced', 'experiences', 'experiment', 'experimental', 'experiments', 'expert', 'explain', 'explained', 'explaining', 'explains', 'explanation', 'explicit', 'exploitation', 'exploration', 'explore', 'explored', 'explosion', 'explosions', 'exposed', 'exposure', 'express', 'expressed', 'expression', 'expressions', 'extended', 'extent', 'extra', 'extraordinary', 'extras', 'extreme', 'extremely', 'eye', 'eyed', 'eyes', 'eyre', 'fabulous', 'face', 'faced', 'faces', 'facial', 'facing', 'fact', 'factor', 'factory', 'facts', 'fail', 'failed', 'failing', 'fails', 'failure', 'fair', 'fairly', 'fairy', 'faith', 'faithful', 'fake', 'falk', 'fall', 'fallen', 'falling', 'falls', 'false', 'fame', 'familiar', 'families', 'family', 'famous', 'fan', 'fancy', 'fans', 'fantastic', 'fantasy', 'far', 'farce', 'fare', 'farm', 'farrell', 'fascinated', 'fascinating', 'fashion', 'fashioned', 'fast', 'faster', 'fat', 'fatal', 'fate', 'father', 'fault', 'faults', 'favor', 'favorite', 'favorites', 'favourite', 'fay', 'fbi', 'fear', 'fears', 'feature', 'featured', 'features', 'featuring', 'fed', 'feed', 'feel', 'feeling', 'feelings', 'feels', 'feet', 'felix', 'fell', 'fellow', 'felt', 'female', 'feminist', 'femme', 'fest', 'festival', 'fetched', 'fever', 'fi', 'fianc', 'fiction', 'fictional', 'fido', 'field', 'fields', 'fifteen', 'fight', 'fighter', 'fighting', 'fights', 'figure', 'figured', 'figures', 'files', 'fill', 'filled', 'film', 'filmed', 'filming', 'filmmaker', 'filmmakers', 'films', 'final', 'finale', 'finally', 'financial', 'find', 'finding', 'finds', 'fine', 'finest', 'finger', 'finish', 'finished', 'fire', 'fired', 'firm', 'first', 'firstly', 'fish', 'fisher', 'fishing', 'fit', 'fits', 'fitting', 'five', 'fix', 'flair', 'flash', 'flashback', 'flashbacks', 'flat', 'flaw', 'flawed', 'flawless', 'flaws', 'flesh', 'flick', 'flicks', 'flies', 'flight', 'floating', 'floor', 'flop', 'florida', 'flow', 'fly', 'flying', 'flynn', 'focus', 'focused', 'focuses', 'focusing', 'folk', 'folks', 'follow', 'followed', 'following', 'follows', 'fond', 'fonda', 'food', 'fool', 'fooled', 'foot', 'footage', 'football', 'forbidden', 'force', 'forced', 'forces', 'ford', 'foreign', 'forest', 'forever', 'forget', 'forgettable', 'forgive', 'forgot', 'forgotten', 'form', 'format', 'former', 'forms', 'formula', 'formulaic', 'forth', 'fortunately', 'fortune', 'forty', 'forward', 'foster', 'fought', 'foul', 'found', 'four', 'fourth', 'fox', 'frame', 'france', 'franchise', 'francis', 'francisco', 'franco', 'frank', 'frankenstein', 'frankie', 'frankly', 'freak', 'fred', 'freddy', 'free', 'freedom', 'freeman', 'french', 'frequent', 'frequently', 'fresh', 'friday', 'friend', 'friendly', 'friends', 'friendship', 'frightening', 'front', 'frustrated', 'frustrating', 'frustration', 'fu', 'fulci', 'full', 'fuller', 'fully', 'fun', 'funeral', 'funnier', 'funniest', 'funny', 'furious', 'furthermore', 'fury', 'future', 'futuristic', 'fx', 'gabriel', 'gadget', 'gag', 'gags', 'gain', 'game', 'games', 'gandhi', 'gang', 'gangster', 'gangsters', 'garbage', 'garbo', 'garden', 'gary', 'gas', 'gave', 'gay', 'gem', 'gender', 'gene', 'general', 'generally', 'generated', 'generation', 'generations', 'generic', 'generous', 'genius', 'genre', 'genres', 'gentle', 'gentleman', 'genuine', 'genuinely', 'george', 'gerard', 'german', 'germans', 'germany', 'get', 'gets', 'getting', 'ghost', 'ghosts', 'giallo', 'giant', 'gift', 'gifted', 'ginger', 'girl', 'girlfriend', 'girls', 'give', 'given', 'gives', 'giving', 'glad', 'glass', 'glasses', 'glenn', 'glimpse', 'global', 'glorious', 'glory', 'glover', 'go', 'goal', 'god', 'godfather', 'godzilla', 'goes', 'going', 'gold', 'goldberg', 'golden', 'gone', 'gonna', 'good', 'goodness', 'goofy', 'gordon', 'gore', 'gorgeous', 'gory', 'got', 'gothic', 'gotta', 'gotten', 'government', 'grab', 'grabs', 'grace', 'grade', 'gradually', 'graham', 'grand', 'grandfather', 'grandmother', 'grant', 'granted', 'graphic', 'graphics', 'grasp', 'gratuitous', 'grave', 'gray', 'grayson', 'great', 'greater', 'greatest', 'greatly', 'greatness', 'greed', 'greedy', 'greek', 'green', 'greg', 'gregory', 'grew', 'grey', 'grief', 'griffith', 'grim', 'grinch', 'gripping', 'gritty', 'gross', 'ground', 'group', 'groups', 'grow', 'growing', 'grown', 'grows', 'gruesome', 'guarantee', 'guard', 'guess', 'guessed', 'guessing', 'guest', 'guide', 'guilt', 'guilty', 'gun', 'gundam', 'guns', 'guts', 'guy', 'guys', 'ha', 'hair', 'hal', 'half', 'halfway', 'hall', 'halloween', 'ham', 'hamilton', 'hamlet', 'hammer', 'hand', 'handed', 'handful', 'handle', 'handled', 'hands', 'handsome', 'hang', 'hanging', 'hank', 'hanks', 'happen', 'happened', 'happening', 'happens', 'happily', 'happiness', 'happy', 'hard', 'hardcore', 'harder', 'hardly', 'hardy', 'harm', 'harris', 'harry', 'harsh', 'hart', 'hartley', 'harvey', 'hat', 'hate', 'hated', 'hates', 'hatred', 'haunted', 'haunting', 'hawke', 'hbo', 'head', 'headed', 'heads', 'health', 'hear', 'heard', 'hearing', 'heart', 'hearted', 'hearts', 'heat', 'heaven', 'heavily', 'heavy', 'heck', 'heights', 'held', 'helen', 'helicopter', 'hell', 'hello', 'help', 'helped', 'helping', 'helps', 'hence', 'henry', 'hero', 'heroes', 'heroic', 'heroine', 'heston', 'hey', 'hidden', 'hide', 'hideous', 'hiding', 'high', 'higher', 'highest', 'highlight', 'highlights', 'highly', 'hilarious', 'hilariously', 'hill', 'hills', 'hint', 'hints', 'hip', 'hippie', 'hire', 'hired', 'historical', 'historically', 'history', 'hit', 'hitchcock', 'hitler', 'hits', 'hitting', 'ho', 'hoffman', 'hold', 'holding', 'holds', 'hole', 'holes', 'holiday', 'hollow', 'holly', 'hollywood', 'holmes', 'holy', 'homage', 'home', 'homeless', 'homer', 'homosexual', 'honest', 'honestly', 'honesty', 'hong', 'honor', 'hood', 'hook', 'hooked', 'hop', 'hope', 'hoped', 'hopefully', 'hopeless', 'hopes', 'hoping', 'hopper', 'horrendous', 'horrible', 'horribly', 'horrid', 'horrific', 'horrifying', 'horror', 'horrors', 'horse', 'horses', 'hospital', 'host', 'hot', 'hotel', 'hour', 'hours', 'house', 'household', 'houses', 'howard', 'however', 'hudson', 'huge', 'hugh', 'huh', 'human', 'humanity', 'humans', 'humble', 'humor', 'humorous', 'humour', 'hundred', 'hundreds', 'hung', 'hunt', 'hunter', 'hunters', 'hunting', 'hurt', 'hurts', 'husband', 'husbands', 'hyde', 'hype', 'hysterical', 'ian', 'ice', 'icon', 'idea', 'ideal', 'ideas', 'identify', 'identity', 'idiot', 'idiotic', 'idiots', 'ignorant', 'ignore', 'ignored', 'ii', 'iii', 'ill', 'illegal', 'illness', 'illogical', 'im', 'image', 'imagery', 'images', 'imagination', 'imaginative', 'imagine', 'imagined', 'imdb', 'imitation', 'immediate', 'immediately', 'immensely', 'impact', 'implausible', 'importance', 'important', 'importantly', 'impossible', 'impress', 'impressed', 'impression', 'impressive', 'improve', 'improved', 'improvement', 'inability', 'inane', 'inappropriate', 'incident', 'include', 'included', 'includes', 'including', 'incoherent', 'incompetent', 'incomprehensible', 'increasingly', 'incredible', 'incredibly', 'indeed', 'independent', 'india', 'indian', 'indians', 'indie', 'individual', 'individuals', 'inducing', 'indulgent', 'industry', 'inept', 'inevitable', 'inevitably', 'infamous', 'inferior', 'influence', 'influenced', 'information', 'ingredients', 'initial', 'initially', 'inner', 'innocence', 'innocent', 'innovative', 'insane', 'inside', 'insight', 'inspector', 'inspiration', 'inspired', 'inspiring', 'installment', 'instance', 'instant', 'instantly', 'instead', 'instinct', 'insult', 'insulting', 'integrity', 'intellectual', 'intelligence', 'intelligent', 'intended', 'intense', 'intensity', 'intent', 'intention', 'intentionally', 'intentions', 'interaction', 'interest', 'interested', 'interesting', 'interests', 'international', 'internet', 'interpretation', 'interview', 'interviews', 'intimate', 'intrigue', 'intrigued', 'intriguing', 'introduce', 'introduced', 'introduces', 'introduction', 'invasion', 'invented', 'inventive', 'investigate', 'investigation', 'invisible', 'involve', 'involved', 'involvement', 'involves', 'involving', 'iran', 'iraq', 'ireland', 'irish', 'iron', 'ironic', 'ironically', 'irony', 'irrelevant', 'irritating', 'island', 'isolated', 'israel', 'issue', 'issues', 'italian', 'italy', 'jack', 'jackie', 'jackson', 'jail', 'jake', 'james', 'jamie', 'jane', 'japan', 'japanese', 'jason', 'jaw', 'jaws', 'jay', 'jazz', 'jealous', 'jean', 'jeff', 'jeffrey', 'jennifer', 'jenny', 'jeremy', 'jerk', 'jerry', 'jesse', 'jessica', 'jesus', 'jet', 'jewish', 'jim', 'jimmy', 'joan', 'job', 'jobs', 'joe', 'joel', 'joey', 'john', 'johnny', 'johnson', 'join', 'joined', 'joke', 'jokes', 'jon', 'jonathan', 'jones', 'joseph', 'josh', 'journalist', 'journey', 'joy', 'jr', 'judge', 'judging', 'judy', 'julia', 'julian', 'julie', 'jump', 'jumped', 'jumping', 'jumps', 'june', 'jungle', 'junior', 'junk', 'justice', 'justify', 'justin', 'juvenile', 'kane', 'kansas', 'kapoor', 'karen', 'karloff', 'kate', 'kay', 'keaton', 'keep', 'keeping', 'keeps', 'keith', 'kelly', 'ken', 'kennedy', 'kenneth', 'kept', 'kevin', 'key', 'khan', 'kick', 'kicked', 'kicking', 'kicks', 'kid', 'kidding', 'kidnapped', 'kids', 'kill', 'killed', 'killer', 'killers', 'killing', 'killings', 'kills', 'kim', 'kind', 'kinda', 'kinds', 'king', 'kingdom', 'kirk', 'kiss', 'kissing', 'kitchen', 'knew', 'knife', 'knock', 'know', 'knowing', 'knowledge', 'known', 'knows', 'kolchak', 'kong', 'korean', 'kubrick', 'kudos', 'kumar', 'kung', 'kurosawa', 'kurt', 'kyle', 'la', 'lab', 'lack', 'lacked', 'lacking', 'lackluster', 'lacks', 'ladies', 'lady', 'laid', 'lake', 'lame', 'land', 'landing', 'landscape', 'landscapes', 'lane', 'language', 'large', 'largely', 'larger', 'larry', 'last', 'lasted', 'late', 'lately', 'later', 'latest', 'latin', 'latter', 'laugh', 'laughable', 'laughably', 'laughed', 'laughing', 'laughs', 'laughter', 'laura', 'laurel', 'law', 'lawrence', 'laws', 'lawyer', 'lay', 'lazy', 'le', 'lead', 'leader', 'leading', 'leads', 'league', 'learn', 'learned', 'learning', 'learns', 'least', 'leave', 'leaves', 'leaving', 'led', 'lee', 'left', 'leg', 'legacy', 'legal', 'legend', 'legendary', 'legs', 'lemmon', 'lena', 'length', 'lengthy', 'leo', 'leon', 'leonard', 'les', 'lesbian', 'leslie', 'less', 'lesser', 'lesson', 'lessons', 'let', 'lets', 'letter', 'letters', 'letting', 'level', 'levels', 'lewis', 'li', 'liberal', 'library', 'lie', 'lies', 'life', 'lifestyle', 'lifetime', 'light', 'lighting', 'lights', 'likable', 'like', 'liked', 'likely', 'likes', 'likewise', 'liking', 'lily', 'limited', 'limits', 'lincoln', 'linda', 'line', 'liners', 'lines', 'link', 'lion', 'lips', 'lisa', 'list', 'listed', 'listen', 'listening', 'lit', 'literally', 'literature', 'little', 'live', 'lived', 'lively', 'lives', 'living', 'lloyd', 'load', 'loaded', 'loads', 'local', 'location', 'locations', 'locked', 'logan', 'logic', 'logical', 'lol', 'london', 'lone', 'lonely', 'long', 'longer', 'look', 'looked', 'looking', 'looks', 'loose', 'loosely', 'lord', 'los', 'lose', 'loser', 'losers', 'loses', 'losing', 'loss', 'lost', 'lot', 'lots', 'lou', 'loud', 'louis', 'louise', 'lousy', 'lovable', 'love', 'loved', 'lovely', 'lover', 'lovers', 'loves', 'loving', 'low', 'lower', 'lowest', 'loyal', 'loyalty', 'lucas', 'luck', 'luckily', 'lucky', 'lucy', 'ludicrous', 'lugosi', 'luke', 'lumet', 'lundgren', 'lust', 'lying', 'lynch', 'lyrics', 'macarthur', 'machine', 'machines', 'macy', 'mad', 'made', 'madness', 'madonna', 'mafia', 'magazine', 'maggie', 'magic', 'magical', 'magnificent', 'maid', 'mail', 'main', 'mainly', 'mainstream', 'maintain', 'major', 'majority', 'make', 'maker', 'makers', 'makes', 'makeup', 'making', 'male', 'mall', 'malone', 'man', 'manage', 'managed', 'manager', 'manages', 'manhattan', 'maniac', 'manipulative', 'mankind', 'mann', 'manner', 'mansion', 'many', 'map', 'marc', 'march', 'margaret', 'maria', 'marie', 'mario', 'marion', 'mark', 'market', 'marketing', 'marks', 'marriage', 'married', 'marry', 'mars', 'marshall', 'martial', 'martin', 'marty', 'marvelous', 'mary', 'mask', 'masks', 'mass', 'massacre', 'masses', 'massive', 'master', 'masterful', 'masterpiece', 'masterpieces', 'masters', 'match', 'matched', 'matches', 'mate', 'material', 'matrix', 'matt', 'matter', 'matters', 'matthau', 'matthew', 'mature', 'max', 'may', 'maybe', 'mayor', 'mclaglen', 'mean', 'meaning', 'meaningful', 'meaningless', 'means', 'meant', 'meanwhile', 'measure', 'meat', 'mechanical', 'media', 'medical', 'mediocre', 'medium', 'meet', 'meeting', 'meets', 'mel', 'melodrama', 'melodramatic', 'melting', 'member', 'members', 'memorable', 'memories', 'memory', 'men', 'menace', 'menacing', 'mental', 'mentally', 'mention', 'mentioned', 'mentioning', 'mentions', 'mere', 'merely', 'merit', 'merits', 'meryl', 'mess', 'message', 'messages', 'messed', 'met', 'metal', 'metaphor', 'method', 'methods', 'mexican', 'mexico', 'mgm', 'michael', 'michelle', 'mickey', 'mid', 'middle', 'midnight', 'might', 'mighty', 'miike', 'mike', 'mild', 'mildly', 'mildred', 'mile', 'miles', 'military', 'milk', 'mill', 'miller', 'million', 'millionaire', 'millions', 'min', 'mind', 'minded', 'mindless', 'minds', 'mine', 'mini', 'minimal', 'minimum', 'minor', 'minute', 'minutes', 'miracle', 'mirror', 'miscast', 'miserable', 'miserably', 'misery', 'miss', 'missed', 'misses', 'missing', 'mission', 'mistake', 'mistaken', 'mistakes', 'mistress', 'mitchell', 'mix', 'mixed', 'mixture', 'miyazaki', 'mm', 'mob', 'model', 'models', 'modern', 'modesty', 'molly', 'mom', 'moment', 'moments', 'mon', 'money', 'monk', 'monkey', 'monkeys', 'monster', 'monsters', 'montage', 'montana', 'month', 'months', 'mood', 'moody', 'moon', 'moore', 'moral', 'morality', 'morgan', 'morning', 'moronic', 'morris', 'mostly', 'mother', 'motion', 'motivation', 'motivations', 'motives', 'mountain', 'mountains', 'mouse', 'mouth', 'move', 'moved', 'movement', 'movements', 'moves', 'movie', 'movies', 'moving', 'mr', 'mrs', 'ms', 'mst', 'mtv', 'much', 'multi', 'multiple', 'mummy', 'mundane', 'murder', 'murdered', 'murderer', 'murderous', 'murders', 'murphy', 'murray', 'museum', 'music', 'musical', 'musicals', 'muslim', 'must', 'myers', 'mysteries', 'mysterious', 'mystery', 'nail', 'naive', 'naked', 'name', 'named', 'namely', 'names', 'nancy', 'narration', 'narrative', 'narrator', 'nasty', 'nathan', 'nation', 'national', 'native', 'natural', 'naturally', 'nature', 'navy', 'nazi', 'nazis', 'nd', 'near', 'nearby', 'nearly', 'neat', 'necessarily', 'necessary', 'neck', 'ned', 'need', 'needed', 'needless', 'needs', 'negative', 'neighbor', 'neighborhood', 'neighbors', 'neil', 'neither', 'nelson', 'neo', 'nephew', 'nerd', 'nervous', 'network', 'never', 'nevertheless', 'new', 'newly', 'newman', 'news', 'newspaper', 'next', 'nice', 'nicely', 'nicholas', 'nicholson', 'nick', 'nicole', 'night', 'nightmare', 'nightmares', 'nights', 'nine', 'ninja', 'niro', 'noble', 'nobody', 'noir', 'noise', 'nominated', 'nomination', 'non', 'none', 'nonetheless', 'nonsense', 'nonsensical', 'normal', 'normally', 'norman', 'north', 'nose', 'nostalgia', 'nostalgic', 'notable', 'notably', 'notch', 'note', 'noted', 'notes', 'nothing', 'notice', 'noticed', 'notion', 'notorious', 'novak', 'novel', 'novels', 'nowadays', 'nowhere', 'nuclear', 'nude', 'nudity', 'number', 'numbers', 'numerous', 'nurse', 'nuts', 'nyc', 'object', 'objective', 'obnoxious', 'obscure', 'obsessed', 'obsession', 'obvious', 'obviously', 'occasion', 'occasional', 'occasionally', 'occur', 'occurred', 'occurs', 'ocean', 'odd', 'oddly', 'odds', 'offended', 'offensive', 'offer', 'offered', 'offering', 'offers', 'office', 'officer', 'officers', 'official', 'often', 'oh', 'oil', 'ok', 'okay', 'old', 'older', 'oliver', 'olivier', 'ollie', 'omen', 'one', 'ones', 'online', 'onto', 'open', 'opened', 'opening', 'opens', 'opera', 'operation', 'opinion', 'opinions', 'opportunities', 'opportunity', 'opposed', 'opposite', 'orange', 'order', 'ordered', 'orders', 'ordinary', 'original', 'originality', 'originally', 'orleans', 'orson', 'oscar', 'oscars', 'othello', 'others', 'otherwise', 'ought', 'outcome', 'outer', 'outfit', 'outrageous', 'outside', 'outstanding', 'overacting', 'overall', 'overcome', 'overdone', 'overlook', 'overlooked', 'overly', 'overrated', 'overwhelming', 'owen', 'owner', 'oz', 'pace', 'paced', 'pacing', 'pacino', 'pack', 'package', 'packed', 'page', 'paid', 'pain', 'painful', 'painfully', 'paint', 'painted', 'painting', 'pair', 'pal', 'palace', 'palance', 'palma', 'paltrow', 'pamela', 'pan', 'panic', 'pants', 'paper', 'par', 'parallel', 'paranoia', 'parent', 'parents', 'paris', 'park', 'parker', 'parody', 'part', 'particular', 'particularly', 'parties', 'partly', 'partner', 'parts', 'party', 'pass', 'passable', 'passed', 'passes', 'passing', 'passion', 'passionate', 'past', 'pat', 'path', 'pathetic', 'patience', 'patient', 'patients', 'patrick', 'paul', 'paulie', 'pay', 'paying', 'pays', 'peace', 'peak', 'pearl', 'people', 'peoples', 'per', 'perfect', 'perfection', 'perfectly', 'perform', 'performance', 'performances', 'performed', 'performer', 'performers', 'performing', 'performs', 'perhaps', 'period', 'perry', 'person', 'persona', 'personal', 'personalities', 'personality', 'personally', 'persons', 'perspective', 'pet', 'pete', 'peter', 'peters', 'petty', 'pg', 'phantom', 'phil', 'philip', 'philosophical', 'philosophy', 'phone', 'phony', 'photo', 'photographed', 'photographer', 'photography', 'photos', 'physical', 'physically', 'piano', 'pick', 'picked', 'picking', 'picks', 'picture', 'pictures', 'pie', 'piece', 'pieces', 'pierce', 'pig', 'pile', 'pilot', 'pin', 'pink', 'pit', 'pitch', 'pitt', 'pity', 'place', 'placed', 'places', 'plague', 'plain', 'plan', 'plane', 'planet', 'planned', 'planning', 'plans', 'plant', 'plastic', 'plausible', 'play', 'played', 'player', 'players', 'playing', 'plays', 'pleasant', 'pleasantly', 'please', 'pleased', 'pleasure', 'plenty', 'plight', 'plot', 'plots', 'plus', 'poem', 'poetic', 'poetry', 'poignant', 'point', 'pointed', 'pointless', 'points', 'pokemon', 'polanski', 'police', 'polished', 'political', 'politically', 'politics', 'pool', 'poor', 'poorly', 'pop', 'popcorn', 'pops', 'popular', 'popularity', 'population', 'porn', 'porno', 'portion', 'portrait', 'portray', 'portrayal', 'portrayed', 'portraying', 'portrays', 'position', 'positive', 'possessed', 'possibilities', 'possibility', 'possible', 'possibly', 'post', 'poster', 'pot', 'potential', 'potentially', 'poverty', 'powell', 'power', 'powerful', 'powers', 'practically', 'practice', 'praise', 'pre', 'precious', 'predictable', 'prefer', 'pregnant', 'premiere', 'premise', 'prepared', 'prequel', 'presence', 'present', 'presentation', 'presented', 'presents', 'president', 'press', 'pressure', 'preston', 'presumably', 'pretend', 'pretending', 'pretentious', 'pretty', 'prevent', 'preview', 'previous', 'previously', 'prey', 'price', 'priceless', 'pride', 'priest', 'primarily', 'primary', 'prime', 'prince', 'princess', 'principal', 'print', 'prior', 'prison', 'prisoner', 'prisoners', 'private', 'prize', 'pro', 'probably', 'problem', 'problems', 'proceedings', 'proceeds', 'process', 'produce', 'produced', 'producer', 'producers', 'producing', 'product', 'production', 'productions', 'professional', 'professor', 'profound', 'program', 'progress', 'progresses', 'project', 'projects', 'prom', 'promise', 'promised', 'promises', 'promising', 'proof', 'propaganda', 'proper', 'properly', 'property', 'props', 'prostitute', 'protagonist', 'protagonists', 'protect', 'proud', 'prove', 'proved', 'proves', 'provide', 'provided', 'provides', 'providing', 'provoking', 'pseudo', 'psychiatrist', 'psychic', 'psycho', 'psychological', 'psychotic', 'public', 'pull', 'pulled', 'pulling', 'pulls', 'pulp', 'punch', 'punishment', 'punk', 'puppet', 'purchase', 'purchased', 'pure', 'purely', 'purple', 'purpose', 'purposes', 'pursuit', 'push', 'pushed', 'pushing', 'put', 'puts', 'putting', 'qualities', 'quality', 'queen', 'quest', 'question', 'questionable', 'questions', 'quick', 'quickly', 'quiet', 'quinn', 'quirky', 'quit', 'quite', 'quote', 'quotes', 'rabbit', 'race', 'rachel', 'racial', 'racism', 'racist', 'radio', 'rage', 'rain', 'raines', 'raise', 'raised', 'raising', 'ralph', 'rambo', 'ramones', 'ran', 'random', 'randomly', 'randy', 'range', 'rangers', 'rank', 'ranks', 'rap', 'rape', 'raped', 'rare', 'rarely', 'rat', 'rate', 'rated', 'rather', 'rating', 'ratings', 'rats', 'rave', 'raw', 'ray', 'raymond', 'rd', 'reach', 'reached', 'reaches', 'reaching', 'react', 'reaction', 'reactions', 'read', 'reader', 'reading', 'reads', 'ready', 'real', 'realise', 'realism', 'realistic', 'reality', 'realize', 'realized', 'realizes', 'realizing', 'really', 'reason', 'reasonable', 'reasonably', 'reasons', 'rebel', 'recall', 'receive', 'received', 'receives', 'recent', 'recently', 'recognition', 'recognize', 'recognized', 'recommend', 'recommended', 'record', 'recorded', 'recording', 'red', 'redeeming', 'redemption', 'reduced', 'reed', 'reel', 'reference', 'references', 'reflect', 'reflection', 'refreshing', 'refused', 'refuses', 'regard', 'regarding', 'regardless', 'regret', 'regular', 'reid', 'relate', 'related', 'relation', 'relations', 'relationship', 'relationships', 'relative', 'relatively', 'relatives', 'release', 'released', 'relevant', 'relief', 'relies', 'religion', 'religious', 'remain', 'remained', 'remaining', 'remains', 'remake', 'remarkable', 'remarkably', 'remarks', 'remember', 'remembered', 'remind', 'reminded', 'reminds', 'reminiscent', 'remote', 'remotely', 'removed', 'rendition', 'rent', 'rental', 'rented', 'renting', 'repeat', 'repeated', 'repeatedly', 'repetitive', 'replaced', 'report', 'reporter', 'represent', 'represented', 'represents', 'reputation', 'required', 'requires', 'rescue', 'research', 'resemblance', 'resemble', 'resembles', 'resident', 'resist', 'resolution', 'resort', 'resources', 'respect', 'respected', 'respectively', 'response', 'responsibility', 'responsible', 'rest', 'restaurant', 'restored', 'result', 'resulting', 'results', 'retarded', 'retired', 'return', 'returned', 'returning', 'returns', 'reunion', 'reveal', 'revealed', 'revealing', 'reveals', 'revelation', 'revenge', 'review', 'reviewer', 'reviewers', 'reviews', 'revolution', 'revolutionary', 'revolves', 'rex', 'reynolds', 'rich', 'richard', 'richards', 'richardson', 'rick', 'rid', 'ridden', 'ride', 'ridiculous', 'ridiculously', 'riding', 'right', 'rights', 'ring', 'rings', 'rip', 'ripped', 'rise', 'rises', 'rising', 'risk', 'rita', 'ritter', 'rival', 'river', 'riveting', 'road', 'rob', 'robbery', 'robbins', 'robert', 'roberts', 'robin', 'robinson', 'robot', 'robots', 'rochester', 'rock', 'rocket', 'rocks', 'rocky', 'roger', 'rogers', 'role', 'roles', 'roll', 'rolled', 'rolling', 'roman', 'romance', 'romantic', 'romero', 'romp', 'ron', 'room', 'rooms', 'rooney', 'root', 'roots', 'rose', 'ross', 'roth', 'rotten', 'rough', 'round', 'routine', 'row', 'roy', 'royal', 'rubber', 'rubbish', 'ruby', 'ruin', 'ruined', 'ruins', 'rukh', 'rule', 'rules', 'run', 'running', 'runs', 'rural', 'rush', 'rushed', 'russell', 'russia', 'russian', 'ruth', 'ruthless', 'ryan', 'sabrina', 'sacrifice', 'sad', 'sadistic', 'sadly', 'sadness', 'safe', 'safety', 'saga', 'said', 'sake', 'sally', 'sam', 'samurai', 'san', 'sandler', 'sandra', 'santa', 'sappy', 'sarah', 'sat', 'satan', 'satire', 'satisfied', 'satisfy', 'satisfying', 'saturday', 'savage', 'save', 'saved', 'saves', 'saving', 'saw', 'say', 'saying', 'says', 'scale', 'scare', 'scarecrow', 'scared', 'scares', 'scary', 'scenario', 'scene', 'scenery', 'scenes', 'scheme', 'school', 'sci', 'science', 'scientific', 'scientist', 'scientists', 'scooby', 'scope', 'score', 'scores', 'scotland', 'scott', 'scottish', 'scream', 'screaming', 'screams', 'screen', 'screening', 'screenplay', 'screenwriter', 'script', 'scripted', 'scripts', 'scrooge', 'sea', 'seagal', 'sean', 'search', 'searching', 'season', 'seasons', 'seat', 'second', 'secondly', 'seconds', 'secret', 'secretary', 'secretly', 'secrets', 'section', 'security', 'see', 'seed', 'seeing', 'seek', 'seeking', 'seeks', 'seem', 'seemed', 'seemingly', 'seems', 'seen', 'sees', 'segment', 'segments', 'seldom', 'self', 'selfish', 'sell', 'sellers', 'selling', 'semi', 'send', 'sends', 'sense', 'senseless', 'senses', 'sensitive', 'sent', 'sentence', 'sentimental', 'separate', 'september', 'sequel', 'sequels', 'sequence', 'sequences', 'serial', 'series', 'serious', 'seriously', 'serve', 'served', 'serves', 'service', 'serving', 'set', 'sets', 'setting', 'settings', 'settle', 'seven', 'seventies', 'several', 'severe', 'sex', 'sexual', 'sexuality', 'sexually', 'sexy', 'sh', 'shadow', 'shadows', 'shake', 'shakespeare', 'shaky', 'shall', 'shallow', 'shame', 'shanghai', 'shape', 'share', 'shark', 'sharp', 'shaw', 'shed', 'sheer', 'shelf', 'shelley', 'sheriff', 'shine', 'shines', 'shining', 'ship', 'ships', 'shirley', 'shirt', 'shock', 'shocked', 'shocking', 'shoes', 'shoot', 'shooting', 'shoots', 'shop', 'short', 'shortly', 'shorts', 'shot', 'shots', 'show', 'showcase', 'showdown', 'showed', 'shower', 'showing', 'shown', 'shows', 'shut', 'shy', 'sick', 'sid', 'side', 'sidekick', 'sides', 'sidney', 'sight', 'sign', 'signed', 'significance', 'significant', 'signs', 'silence', 'silent', 'silly', 'silver', 'similar', 'similarities', 'similarly', 'simmons', 'simon', 'simple', 'simplicity', 'simplistic', 'simply', 'simpson', 'sin', 'sinatra', 'since', 'sincere', 'sing', 'singer', 'singers', 'singing', 'single', 'sings', 'sinister', 'sink', 'sir', 'sirk', 'sissy', 'sister', 'sisters', 'sit', 'sitcom', 'site', 'sits', 'sitting', 'situation', 'situations', 'six', 'sixties', 'size', 'skill', 'skills', 'skin', 'skip', 'skull', 'sky', 'slap', 'slapstick', 'slasher', 'slaughter', 'slave', 'sleazy', 'sleep', 'sleeping', 'slice', 'slick', 'slight', 'slightest', 'slightly', 'sloppy', 'slow', 'slowly', 'small', 'smaller', 'smart', 'smile', 'smiling', 'smith', 'smoke', 'smoking', 'smooth', 'snake', 'sneak', 'snl', 'snow', 'soap', 'soccer', 'social', 'society', 'soderbergh', 'soft', 'sold', 'soldier', 'soldiers', 'sole', 'solely', 'solid', 'solo', 'solution', 'solve', 'somebody', 'somehow', 'someone', 'something', 'sometimes', 'somewhat', 'somewhere', 'son', 'song', 'songs', 'sons', 'soon', 'sophisticated', 'sopranos', 'sorry', 'sort', 'sorts', 'soul', 'souls', 'sound', 'sounded', 'sounding', 'sounds', 'soundtrack', 'source', 'south', 'southern', 'soviet', 'space', 'spacey', 'spain', 'spanish', 'spare', 'spark', 'speak', 'speaking', 'speaks', 'special', 'specially', 'species', 'specific', 'specifically', 'spectacular', 'speech', 'speed', 'spell', 'spend', 'spending', 'spends', 'spent', 'spider', 'spielberg', 'spike', 'spin', 'spirit', 'spirited', 'spirits', 'spiritual', 'spite', 'splatter', 'splendid', 'split', 'spock', 'spoil', 'spoiled', 'spoiler', 'spoilers', 'spoke', 'spoken', 'spoof', 'spooky', 'sport', 'sports', 'spot', 'spots', 'spread', 'spring', 'spy', 'square', 'st', 'stack', 'staff', 'stage', 'staged', 'stale', 'stallone', 'stan', 'stand', 'standard', 'standards', 'standing', 'stands', 'stanley', 'stanwyck', 'star', 'stargate', 'staring', 'starred', 'starring', 'stars', 'start', 'started', 'starting', 'starts', 'state', 'stated', 'statement', 'states', 'station', 'status', 'stay', 'stayed', 'staying', 'stays', 'steal', 'stealing', 'steals', 'steel', 'stellar', 'step', 'stephen', 'steps', 'stereotype', 'stereotypes', 'stereotypical', 'steve', 'steven', 'stevens', 'stewart', 'stick', 'sticks', 'stiff', 'still', 'stiller', 'stilted', 'stinker', 'stinks', 'stock', 'stole', 'stolen', 'stomach', 'stone', 'stood', 'stooges', 'stop', 'stopped', 'stops', 'store', 'stories', 'storm', 'story', 'storyline', 'storytelling', 'straight', 'strange', 'strangely', 'stranger', 'strangers', 'streep', 'street', 'streets', 'streisand', 'strength', 'stress', 'stretch', 'stretched', 'strictly', 'strike', 'strikes', 'striking', 'string', 'strip', 'strong', 'stronger', 'strongly', 'struck', 'structure', 'struggle', 'struggles', 'struggling', 'stuart', 'stuck', 'student', 'students', 'studio', 'studios', 'study', 'stuff', 'stumbled', 'stunned', 'stunning', 'stunt', 'stunts', 'stupid', 'stupidity', 'style', 'styles', 'stylish', 'sub', 'subject', 'subjected', 'subjects', 'subplot', 'subplots', 'subsequent', 'substance', 'subtitles', 'subtle', 'subtlety', 'succeed', 'succeeded', 'succeeds', 'success', 'successful', 'successfully', 'suck', 'sucked', 'sucks', 'sudden', 'suddenly', 'sue', 'suffer', 'suffered', 'suffering', 'suffers', 'suffice', 'suggest', 'suggested', 'suggests', 'suicide', 'suit', 'suitable', 'suited', 'suits', 'sullivan', 'sum', 'summary', 'summer', 'sun', 'sunday', 'sunshine', 'super', 'superb', 'superbly', 'superficial', 'superhero', 'superior', 'superman', 'supernatural', 'support', 'supported', 'supporting', 'suppose', 'supposed', 'supposedly', 'sure', 'surely', 'surface', 'surfing', 'surprise', 'surprised', 'surprises', 'surprising', 'surprisingly', 'surreal', 'surrounded', 'surrounding', 'survival', 'survive', 'survived', 'surviving', 'survivor', 'survivors', 'susan', 'suspect', 'suspects', 'suspend', 'suspense', 'suspenseful', 'suspicious', 'sutherland', 'swear', 'swedish', 'sweet', 'swim', 'swimming', 'switch', 'sword', 'symbolism', 'sympathetic', 'sympathy', 'synopsis', 'system', 'table', 'tacky', 'tad', 'tag', 'take', 'taken', 'takes', 'taking', 'tale', 'talent', 'talented', 'talents', 'tales', 'talk', 'talked', 'talking', 'talks', 'tall', 'tame', 'tank', 'tap', 'tape', 'tarantino', 'target', 'tarzan', 'task', 'taste', 'taught', 'taxi', 'taylor', 'tea', 'teach', 'teacher', 'teaching', 'team', 'tear', 'tears', 'tech', 'technical', 'technically', 'technicolor', 'technique', 'techniques', 'technology', 'ted', 'tedious', 'teen', 'teenage', 'teenager', 'teenagers', 'teens', 'teeth', 'television', 'tell', 'telling', 'tells', 'temple', 'ten', 'tend', 'tender', 'tends', 'tense', 'tension', 'term', 'terms', 'terrible', 'terribly', 'terrific', 'terrifying', 'territory', 'terror', 'terrorist', 'terrorists', 'terry', 'test', 'testament', 'texas', 'text', 'th', 'thank', 'thankfully', 'thanks', 'thats', 'theater', 'theaters', 'theatre', 'theatrical', 'theme', 'themes', 'theory', 'therefore', 'thick', 'thief', 'thin', 'thing', 'things', 'think', 'thinking', 'thinks', 'third', 'thirty', 'thomas', 'thompson', 'thoroughly', 'though', 'thought', 'thoughtful', 'thoughts', 'thousand', 'thousands', 'threat', 'threatening', 'three', 'threw', 'thrill', 'thriller', 'thrillers', 'thrilling', 'thrills', 'throat', 'throughout', 'throw', 'throwing', 'thrown', 'throws', 'thru', 'thugs', 'thumbs', 'thus', 'ticket', 'tie', 'tied', 'ties', 'tiger', 'tight', 'till', 'tim', 'time', 'timeless', 'times', 'timing', 'timon', 'timothy', 'tiny', 'tired', 'tiresome', 'titanic', 'title', 'titled', 'titles', 'today', 'todd', 'together', 'toilet', 'told', 'tom', 'tomatoes', 'tommy', 'tone', 'tongue', 'tonight', 'tons', 'tony', 'took', 'tooth', 'top', 'topic', 'topless', 'torn', 'torture', 'tortured', 'total', 'totally', 'touch', 'touched', 'touches', 'touching', 'tough', 'tour', 'toward', 'towards', 'town', 'toy', 'toys', 'track', 'tracks', 'tracy', 'trade', 'trademark', 'tradition', 'traditional', 'tragedy', 'tragic', 'trail', 'trailer', 'trailers', 'train', 'trained', 'training', 'transfer', 'transformation', 'transition', 'translation', 'trap', 'trapped', 'trash', 'trashy', 'travel', 'traveling', 'travels', 'travesty', 'treasure', 'treat', 'treated', 'treatment', 'treats', 'tree', 'trees', 'trek', 'tremendous', 'trial', 'tribe', 'tribute', 'trick', 'tricks', 'tried', 'tries', 'trilogy', 'trio', 'trip', 'trite', 'triumph', 'troops', 'trouble', 'troubled', 'troubles', 'truck', 'true', 'truly', 'trust', 'truth', 'try', 'trying', 'tube', 'tune', 'tunes', 'turkey', 'turn', 'turned', 'turner', 'turning', 'turns', 'tv', 'twelve', 'twenty', 'twice', 'twilight', 'twin', 'twins', 'twist', 'twisted', 'twists', 'two', 'type', 'types', 'typical', 'typically', 'ugly', 'uk', 'ultimate', 'ultimately', 'ultra', 'un', 'unable', 'unaware', 'unbearable', 'unbelievable', 'unbelievably', 'uncle', 'uncomfortable', 'unconvincing', 'underground', 'underlying', 'underrated', 'understand', 'understandable', 'understanding', 'understated', 'understood', 'undoubtedly', 'uneven', 'unexpected', 'unexpectedly', 'unfair', 'unfolds', 'unforgettable', 'unfortunate', 'unfortunately', 'unfunny', 'unhappy', 'uninspired', 'unintentional', 'unintentionally', 'uninteresting', 'union', 'unique', 'unit', 'united', 'universal', 'universe', 'university', 'unknown', 'unless', 'unlike', 'unlikely', 'unnecessary', 'unoriginal', 'unpleasant', 'unpredictable', 'unreal', 'unrealistic', 'unseen', 'unsettling', 'unusual', 'unwatchable', 'uplifting', 'upon', 'upper', 'ups', 'upset', 'urban', 'urge', 'us', 'usa', 'use', 'used', 'useful', 'useless', 'user', 'uses', 'using', 'ustinov', 'usual', 'usually', 'utter', 'utterly', 'uwe', 'vacation', 'vague', 'vaguely', 'valentine', 'valley', 'valuable', 'value', 'values', 'vampire', 'vampires', 'van', 'vance', 'variety', 'various', 'vast', 'vegas', 'vehicle', 'vengeance', 'verhoeven', 'version', 'versions', 'versus', 'veteran', 'vhs', 'via', 'vice', 'vicious', 'victim', 'victims', 'victor', 'victoria', 'victory', 'video', 'videos', 'vietnam', 'view', 'viewed', 'viewer', 'viewers', 'viewing', 'viewings', 'views', 'village', 'villain', 'villains', 'vincent', 'violence', 'violent', 'virgin', 'virginia', 'virtually', 'virus', 'visible', 'vision', 'visit', 'visits', 'visual', 'visually', 'visuals', 'vivid', 'voice', 'voiced', 'voices', 'voight', 'von', 'vote', 'vs', 'vulnerable', 'wacky', 'wait', 'waited', 'waiting', 'waitress', 'wake', 'walk', 'walked', 'walken', 'walker', 'walking', 'walks', 'wall', 'wallace', 'walls', 'walsh', 'walter', 'wandering', 'wang', 'wanna', 'wannabe', 'want', 'wanted', 'wanting', 'wants', 'war', 'ward', 'warm', 'warming', 'warmth', 'warn', 'warned', 'warner', 'warning', 'warren', 'warrior', 'warriors', 'wars', 'washington', 'waste', 'wasted', 'wasting', 'watch', 'watchable', 'watched', 'watches', 'watching', 'water', 'waters', 'watson', 'wave', 'waves', 'way', 'wayne', 'ways', 'weak', 'weakest', 'wealth', 'wealthy', 'weapon', 'weapons', 'wear', 'wearing', 'wears', 'web', 'website', 'wedding', 'week', 'weekend', 'weeks', 'weight', 'weird', 'welcome', 'well', 'welles', 'wells', 'wendy', 'went', 'werewolf', 'wes', 'west', 'western', 'westerns', 'wet', 'whale', 'whatever', 'whats', 'whatsoever', 'whenever', 'whereas', 'whether', 'whilst', 'white', 'whoever', 'whole', 'wholly', 'whoopi', 'whose', 'wicked', 'wide', 'widely', 'widmark', 'widow', 'wife', 'wild', 'wilderness', 'william', 'williams', 'willie', 'willing', 'willis', 'wilson', 'win', 'wind', 'window', 'winds', 'wing', 'winner', 'winning', 'wins', 'winter', 'winters', 'wisdom', 'wise', 'wish', 'wished', 'wishes', 'wishing', 'wit', 'witch', 'witches', 'within', 'without', 'witness', 'witnessed', 'witnesses', 'witty', 'wives', 'wizard', 'wolf', 'woman', 'women', 'wonder', 'wondered', 'wonderful', 'wonderfully', 'wondering', 'wonders', 'wont', 'woo', 'wood', 'wooden', 'woods', 'woody', 'word', 'words', 'wore', 'work', 'worked', 'worker', 'workers', 'working', 'works', 'world', 'worlds', 'worn', 'worried', 'worry', 'worse', 'worst', 'worth', 'worthless', 'worthwhile', 'worthy', 'would', 'wound', 'wounded', 'wow', 'wrap', 'wrapped', 'wreck', 'wrestling', 'wretched', 'write', 'writer', 'writers', 'writes', 'writing', 'written', 'wrong', 'wrote', 'ww', 'wwii', 'ya', 'yard', 'yeah', 'year', 'years', 'yelling', 'yellow', 'yes', 'yesterday', 'yet', 'york', 'young', 'younger', 'youth', 'zero', 'zizek', 'zombie', 'zombies', 'zone']


## 3 建模并测试

### 3.1 建立模型

这里选择随机森林模型，参数介绍以及调参方法见：Sklearn-RandomForest随机森林https://blog.csdn.net/cherdw/article/details/54971771


```python
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit(train_data_features,train["sentiment"])
```

    Training the random forest...


### 3.2 模型测试

(1)模型测试之前，需要准备测试数据，以及提取测试数据的特征


```python
# 读取测试数据
test=pd.read_csv("./data/testData.tsv", header=0, delimiter='\t',quoting=3)
print(test.shape)
# test.head()
```

    (25000, 2)



```python
# 对测试数据进行清洗
num_reviews=len(test['review'])
clean_test_reviews=[]

print("cleaning and parsing the test set movie reviews ..\n")
for i in range(0,num_reviews):
    if((i+1)%1000==0):
        print("Review %d of %d \n"%(i+1,num_reviews))
    clean_review=review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
```

    cleaning and parsing the test set movie reviews ..
    


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
    



```python
# 提取测试集数据的特征
test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()
```

(2)预测


```python
result=forest.predict(test_data_features)
```

(3)整理输出数据格式


```python
output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
output.to_csv("Bag_of_Words_modle.csv",index=False,quoting=3)
```
