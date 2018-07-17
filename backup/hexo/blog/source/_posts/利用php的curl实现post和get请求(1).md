---
title: 利用php的curl实现post和get请求
date: 2018-06-08 21:44:26
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/php.jpg
tags: 
      - 实验教程
      - PHP
categories: [Experiment,PHP]
---
由于项目需求，进行了一段时间的微信开发。在微信开发的过程中，经常调用微信接口，通常是向微信服务器发送get或者post请求获取接口。下面给出两个具体的实现过程。
<!--more-->
#### 1、发送post请求

```
function curl_post($url,$data){ 
    $curl = curl_init(); 
    curl_setopt($curl, CURLOPT_URL, $url); 
    curl_setopt($curl, CURLOPT_SSL_VERIFYPEER, 0); 
    curl_setopt($curl, CURLOPT_SSL_VERIFYHOST, 2); 
    curl_setopt($curl, CURLOPT_USERAGENT, $_SERVER['HTTP_USER_AGENT']); 
    curl_setopt($curl, CURLOPT_FOLLOWLOCATION, 1); 
    curl_setopt($curl, CURLOPT_AUTOREFERER, 1); 
    curl_setopt($curl, CURLOPT_POST, 1); 
    curl_setopt($curl, CURLOPT_POSTFIELDS, $data); 
    curl_setopt($curl, CURLOPT_TIMEOUT, 30); 
    curl_setopt($curl, CURLOPT_HEADER, 0); 
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, 1); 
    $tmpInfo = curl_exec($curl); 
    if (curl_errno($curl)) {
       echo 'Errno'.curl_error($curl);
    }
    curl_close($curl); 
    return $tmpInfo; 
}
```
其中$url是调用微信接口的url，$data是向微信服务器发送的数据，下同。

#### 2、发送get请求

```
function http_post($url,$data)
		{
//			$data_string=json_encode(array('name'=>$name,'data'=>$data));
//			$url = "http://192.168.1.40:8080/wechatdemo/";
			$ch = curl_init($url);
			curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");
			curl_setopt($ch, CURLOPT_POSTFIELDS, $data);//$data JSON类型字符串
			curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
			curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json', 'Content-Length: ' . strlen($data)));
			$result = curl_exec($ch);
			var_dump($result);
		}
```
