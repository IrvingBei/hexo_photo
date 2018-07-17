---
title: C++中调用python程序
date: 2017-12-20 15:41:32
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/c.jpg
tags: 
      - 实验教程
      - C++
categories: [Experiment,C++]
---
> 因项目需要，需要在c++中调用python文件，于是在网上查询相关资料，终于实现，大概搞了大半天，所以还是记录下这个过程
<!--more-->
### 1 相关介绍
（1）基本情况
++在C/C++中嵌入Python，可以使用Python提供的强大功能，通过嵌入Python可以替代动态链接库形式的接口，这样可以方便地根据需要修改脚本代码，而不用重新编译链接二进制的动态链接库。至少你可以把它当成文本形式的动态链接库，需要的时候还可以改一改，只要不改变接口， C++的程序一旦编译好了，再改就没那么方便了。
（2）C++调用Python有两种方式
第一种方式：通过找到Python模块，类，方法，构造参数来调用。
第二中方式，就是通过构造出一个Python的脚本，用python引擎来执行。
第一种方式可能更为优雅，符合大多数的反射调用的特点。（如c#的反射机制，c#调用Com+，c#调用javascript脚本等）。
一个问题：两种语言互相调用的时候，需要做数据结构（如基本类型，字符串，整数类型等，以及自定义的类等类型）间的转换，共享内存中的一个对象。比如，如何将C++的对象实例传入python中，并在python中使用。c++和python并不在一个进程中，因此可以使用boost的shared_ptr来实现。Python调用C++，换句话说就是需要把C++封装成Python可以“理解”的类型。同理可知C++怎么去调用Python脚本。
下面这个例子，主要是演示了c++调用python，可以在c++中形成一个python脚本，然后利用PyRun_SimpleString调用;并且，构造一个c++的对象，传入到python中，并在python的脚本中调用其函数。

### 2 实验环境
（1）vs2017
具体配置见 [vs安装配置](http://blog.csdn.net/pipisorry/article/details/49532341)
（2）clion

配置是修改cmakelist文件，添加：
```
include_directories(/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7)
set(CMAKE_LIBRARY_PATH "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/")
link_libraries(python)
```
### 3 编写程序
（1）方式一：，通过构造出一个Python的脚本，用python引擎来执行。
①主程序

```
#include "stdafx.h"
#include<Python.h>//前面所做的一切配置都是为了调用这个头文件和相关库
#include<iostream>
using namespace std;
/**g++ -o callpy callpy.cpp -I/usr/include/python2.6 -L/usr/lib64/python2.6/config -lpython2.6**/
int main(int argc, char** argv)
{
	// 初始化Python  
	//在使用Python系统前，必须使用Py_Initialize对其  
	//进行初始化。它会载入Python的内建模块并添加系统路  
	//径到模块搜索路径中。这个函数没有返回值，检查系统  
	//是否初始化成功需要使用Py_IsInitialized。  
	Py_Initialize();

	// 检查初始化是否成功  
	if (!Py_IsInitialized()) {
		return -1;
	}
	// 添加当前路径  
	//把输入的字符串作为Python代码直接运行，返回0  
	//表示成功，-1表示有错。大多时候错误都是因为字符串  
	//中有语法错误。  
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("print('---import sys---')");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("import pytest");
	PyRun_SimpleString("pytest.add()");
	Py_Finalize();    
	system("pause");
	return 0;
}
```
②python程序：pytest.py

```
#test function  
def add():  
    print("hello world")
  
```
运行结果：
```
---import sys---
hello world
请按任意键继续. . .
```
（2）方式二：通过找到Python模块，类，方法，构造参数来调用。
这中方法我的电脑上没实验成功，我同学实验成功了。主要参照如下教程的代码：
①[浅析 C++ 调用 Python 模块](https://www.cnblogs.com/findumars/p/6142330.html)
② [C++调用python](http://blog.csdn.net/pipisorry/article/details/49532341)
③[Python实例浅谈之三Python与C/C++相互调用](https://www.cnblogs.com/apexchu/p/5015961.html)

### 4 相关注意
（1）pystring_fromstring没有定义
造成这个问题的原因是：python3+中没有这个函数，所以这个实验只能在python2+的环境下进行。
### 5 相关参考
（1）[浅析 C++ 调用 Python 模块](https://www.cnblogs.com/findumars/p/6142330.html)
（2） [C++调用python](http://blog.csdn.net/pipisorry/article/details/49532341)
（3）[Python实例浅谈之三Python与C/C++相互调用](https://www.cnblogs.com/apexchu/p/5015961.html)