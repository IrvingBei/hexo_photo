
---
title: scrapy-01：简单实列
date: 2017-12-16 09:20:00
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/pachong.jpg
tags: scrapy
categories: [Experiment,Scrapy]
---
> 最近开始学习自然语言处理，在自然语言处理的过程中，获取语料，处理语料都是非常重要的一部，因此在开始入门学习自然语言处理的同时，也开始练习一些爬虫，暑假时学过一点，但是也忘得差不多了，于是重新开始，并记录下来、
<!--more-->


#### 1  scrapy库介绍
（1）具体的介绍可以自行百度

（2）安装方法：
```
#利用pip安装
pip install scrapy
#利用conda安装
conda install scrapy
```
#### 2 scrapy的简单使用
（1）代码实例

```

# coding: utf-8

import scrapy
from w3lib.html import remove_tags

#定义一个类，继承了scrapy，spider
class StackOvverflowSpider(scrapy.Spider):
   #爬虫项目的名字，在整个项目中，这个名字需要唯一
   name="stackoverflow"
   #指定爬虫开始的网址链接
   start_urls=["http://stackoverflow.com/questions?sort=votes"]
   
   #爬虫项目的回调函数
   def parse(self,response):
       #通过css样式来获取需要遍历的链接，
       for href in response.css(".question-summary h3 a::attr(href)"):
           #组装成完整的链接
           full_url=response.urljoin(href.extract())
           #对每一个完整的链接，传给回调函数，在回调函数中抓取页面内容
           yield scrapy.Request(full_url,callback=self.parse_question)
   
   
   def parse_question(self,response):
       yield{
           #指定要抓取的内容，title是抓取结果中保存的字段，后面对应的是他的页面上的内容，后面的css中的类名，需要事先打开网页源代码查看。
           'title':response.css('h1 a::text').extract()[0],
           'votes':response.css('.question .vote-count-post::text').extract()[0],
           'body':response.css('.question .post-text').extract()[0]
       }


```
（2）运行代码

```
scrapy runspider  scrapy_1.py -o abc.csv
```
（3）代码讲解：见代码中的注释

