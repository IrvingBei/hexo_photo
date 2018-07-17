
---
title: scrapy-02：创建工程项目的简单步骤
date: 2017-12-16 20:09:40
thumbnail:  https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/pachong.jpg
tags: scrapy
categories: [Experiment,Scrapy]
---
前面看了一个爬虫的小例子，那么接下来就；来看看创建一个爬虫的步骤有哪些。
<!--more-->
### 1 创建工程项目的简单步骤
（1）开始一个爬虫项目
进入一个你要创建爬虫的目录下面，在命令行窗口中执行如下代码：

```
scrapy startproject 项目名
```
其中项目名改成你将要创建的项目的名称
项目创建好了以后，进入该项目，可以看到如下的目录结构（其中tutorial是创建的项目名）
![image](https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/QQ%E5%9B%BE%E7%89%8720171216191210.png)
（2）创建一个spider文件模板
首先进入项目文件里，在项目文件里用命令行执行

```
scrapy genspider abc_spider www.baidu.com
```
命令行的第三个串是要创建的spider的名字，第四个是要爬取网页的链接地址。
这时，进入项目文件下的spider目录里面，就会发现一个abc_spider.py文件，这个就是创建的文件模板，用IDE打开，修改其内容。

```
# -*- coding: utf-8 -*-
import scrapy


class AbcSpiderSpider(scrapy.Spider):
    name = 'abc_spider'
    allowed_domains = ['www.baidu.com']
    start_urls = ['http://www.baidu.com/']

    def parse(self, response):
        #定义文件名
        filename=response.url.split('/')[-2]+"html"
        #打开文件并写入源代码
        with open(filename,"wb") as fp:
            fp.write(response.body)


```
（3）运行爬虫

```
scrapy crawl 爬虫名
```
于是就能看到一个html的文件。

### 2 自定义抓取项
（1）打开项目中的item.py文件，按照说明进行改写。

```
class FirstspiderItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()
    desc = scrapy.Field()
    link=scrapy.Field()
```
（2）改写parse函数

```
# -*- coding: utf-8 -*-
import scrapy
from firstspider.items import FirstspiderItem

class AbcSpiderSpider(scrapy.Spider):
    name = 'abc_spider'
    allowed_domains = ['www.baidu.com']
    start_urls = ['http://www.baidu.com/']

    def parse(self, response):
        # #定义文件名
        # filename=response.url.split('/')[-2]+"html"
        # #打开文件并写入源代码
        # with open(filename,"wb") as fp:
        #     fp.write(response.body)
        #通过xpath获取要抓取的内容
        lis=response.xpath('//*[@id="head"]')
        for li in lis:
            #在这之前需要引入
            item=FirstspiderItem()
            #xpath百度首页不好举例
            item['title']=li.xpath('xpath路径').extract()
            item['link']=li.xpath('xpath路径').extract()
            item['desc'] = li.xpath('xpath路径').extract()
            #返回item
            yield item

```
（3）改写pipeline文件
这里不做修改，只是返回item即可。

```
class FirstspiderPipeline(object):
    def process_item(self, item, spider):
        return item
```
（4）运行爬虫
可以通过

```
scrapy list
```
来获取项目里面所包含的爬虫，然后运行你要执行的爬虫。抓取得结果一般情况下是存放在数据库中。
