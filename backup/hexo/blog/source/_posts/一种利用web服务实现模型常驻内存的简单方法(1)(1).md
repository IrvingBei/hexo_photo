---
title: 一种利用web服务实现模型常驻内存的简单方法
date: 2018-06-09 15:53:05
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/1.jpg
tags: 
      - 实验教程
      - python
categories: [Experiment,Python]
---

利用python的web服务快速实现模型常驻内存，本方法很low，但是真的很快速，半个小时都能实现。<!--more-->

------
### 1、背景
前段时间有这样一个需求，通过php接收微信服务器发送来的消息，然后把消息发送给python功能逻辑处理程序。在前期是直接使用php调用python程序，python程序进行模型加载，消息处理，然后返回给php。利用这种流程，处理过程不慢都难。但是还必须得按照这样的流程走，于是想想能不能将之前训练好的模型常驻内存。查阅网上，解决方法有很多，可以利用socket通信、做成客户/服务器模式等，这些方法都非常好，但是实现起来有一定的复杂。所以想到了这种方法。
### 2、整体思路
主要是用python以及其flask库来实现的。主要思路有以下几点：
> * 将php调用python程序改为php通过get或post方法向python发送请求；
> * python程序改成web服务模式，运行的就加载模型，让它一直运行；
> * python接收请求，处理，并返回结果；

Flask是一个使用 Python 编写的轻量级 Web 应用框架，使用时，只需要在python中引入即可，因为需要处理get或者post请求，同时引入request，接下来就按照流程来编写代码即可。主要步骤如下：
(1)flask的简单实验
* 引入包创建一个flask应用
```
from flask import Flask,request
app = Flask(__name__)
```
* 定义一个方法
```
def hello():
        print("hello world")
```
* 运行应用
```
app.debug = True
app.run()
```
完整代码：
```
from flask import Flask,request
app = Flask(__name__)

# 加载模型

# 处理请求
@app.route('/hello')
def hello():
    return("hello world")

if __name__ == '__main__':
    app.debug = True
    app.run()
```
此时运行该程序，在浏览器地址栏输入：127.0.0.1:5000/hello，此时浏览器中就会显示hello world，好了，那么接下来的事情就简单了。只需要把上面加载模型和处理请求部分的代码稍作修改就可以了。

* 加载模型
```
model= load_model_to_memory()  # 原来加载模型的那些代码
```
* 接收请求并处理
```
# 处理请求
@app.route('/deal', methods=['GET'])
def deal():
        # 获取php发来的消息
        question = request.args.get('question',"default question")   # 键值 默认值
        # 对消息进行解码
        question=urllib.parse.unquote(question)
        result = main_function(model)  # 该函数就是原来的主要功能逻辑处理函数
        # 返回处理结果
        return (urllib.parse.quote(result))
```
（2）php与python之间通信
上面的程序一直运行着，于是可以通过该url地址向python程序发送请求。在php中构造get或者post请求的方法可以见另一篇文章[利用php的curl实现post和get请求](http://bei.dreamcykj.com/2018/06/08/%E5%88%A9%E7%94%A8php%E7%9A%84curl%E5%AE%9E%E7%8E%B0post%E5%92%8Cget%E8%AF%B7%E6%B1%82%281%29/%22%E5%88%A9%E7%94%A8php%E7%9A%84curl%E5%AE%9E%E7%8E%B0post%E5%92%8Cget%E8%AF%B7%E6%B1%82%22) ，这样就实现了php与python之间的通信，如果是在linux中，可以创建一个tmux会话窗口来运行上面的那个python程序，一直运行下去。
### 3、完整代码
* PHP
```
// 通过get方式将问题传给python的web服务
	public static function send_question_to_python_get($request)
	{
		//本地接收问题的python服务的url
		$base_url="http://127.0.0.1:5000/login?question=";
		// 获取问题，并将其编码
		$question=$request['content'];
		$en_question=urlencode($question);
		$url=$base_url.$en_question;

		$ch = curl_init();
    	//设置选项，包括URL
    	curl_setopt($ch, CURLOPT_URL, $url);
    	curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
    	curl_setopt($ch, CURLOPT_HEADER, 0);
    	curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);//绕过ssl验证
    	curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);
    	//执行并获取HTML文档内容
    	$output = curl_exec($ch);
		$output=urldecode($output);
    	//释放curl句柄
    	curl_close($ch);
    	return $output;
	}
```

* Python
```
from flask import Flask,url_for,request
import urllib.parse
app = Flask(__name__)

# 将模型加载到内存
model = load_model_to_memory()

# 处理请求
@app.route('/deal', methods=['GET'])
def deal():
        # 获取php发来的消息
        question = request.args.get('question',"default question")   # 键值 默认值
        # 对消息进行解码
        question=urllib.parse.unquote(question)
        result = main_function(model)  # 该函数就是原来的主要功能逻辑处理函数
        # 返回处理结果
        return (urllib.parse.quote(result))

if __name__ == '__main__':
    app.debug = True
    app.run()
```

### 4、相关链接
【1】[Flask快速入门](https://blog.csdn.net/u011054333/article/details/70151857)
【2】[利用php的curl实现post和get请求](http://bei.dreamcykj.com/2018/06/08/%E5%88%A9%E7%94%A8php%E7%9A%84curl%E5%AE%9E%E7%8E%B0post%E5%92%8Cget%E8%AF%B7%E6%B1%82%281%29/)

