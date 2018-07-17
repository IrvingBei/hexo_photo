---
title: 【NLTK基础教程】02  何为文本歧义
date: 2018-1-22 20:18:37
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/nltk.jpg
tags: NLTK基础教程
categories: [NLP,NLTK]
---
文本歧义，书中的定义式从原生数据中获取一段机器可读的已经格式化文本之前所要做的所有预处理工作，以及所有繁复的任务。该过程涉及到数据再加工，文本清理，特定项处理，标识化处理，词干提取或词型还原以及停用词移除等操作。
<!--more-->
好吧，书中将文本歧义定义为数据预处理这一些列工作，难到文本歧义不应该是一个文本，多个意思，从而有歧义这个意思吗？没搞懂（问号脸.jpg）.如果是把文本歧义理解成通过一系列数据预处理工作，消除文本歧义，好吧，貌似说得通。那也不纠结那么多了，下面看一个例子，解析一个csv文件。

```
import csv
with open('example.csv') as f:
    reader=csv.reader(f,delimiter=',',quotechar='"')
    for line in reader:
        print line[1]
```
代码说明：
这几句代码整体上是没有什么问题的，这里只是提下csv的reader方法的参数：
①delimiter：一行中的分隔符
②quotechar：每个字段用的类型符号
这里就会涉及到处理文档类型的一般流程，具体见下图：
![image](http://img.blog.csdn.net/20180122210720461?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveHl6MTU4NDE3MjgwOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
在大多数情况下，我们所遇到的这些数据中的某一个，而python中也有对于这些数据格式最常见的封装格式。通过该模块，我们可以使用各种不同的分离器和引用符等工具。
接下来，我们再来看一个json文件示例：
①json数据为：

```
{
	"array":[1,2,3,4],
	"boolean":True,
	"object":{
		"a":"b"
	},
	"string":"hello world"
}

```
②处理该字符串的解析代码如下：

```
import json
jsonfile=open("example.json")
data=json.load(jsonfile)
print(data['string'])
```
好吧，这个就记录到这里吧！