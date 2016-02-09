---
layout: default
title: Github创建个人主页笔记
---

# {{ page.title }}

## 安装Jekyll

1. 安装ruby环境

	```
    sudo apt-get install ruby1.9.1-dev
    ```
    安装完成后在终端输入ruby-v，出现如下结果说明安装成功：
    ```
    ruby 1.9.3p484 (2013-11-22 revision 43786) [x86_64-linux]
    ```
2. 安装jekyll环境

	```
    sudo apt-get install jekyll
    ```
    安装完成后在终端输入如下命令，如果成功创建目录，则说明jekyll安装成功：
    ```
    jekyll new myblog
    ```

## 运行本地工程

进入到工程目录下，启动服务：
```
jekyll --server
```
预览http://127.0.0.1:4000

## jekyll目录结构

- `_post`：用来存放文章，一般以日期形式写标题。
- `_layouts`：用来存放模板，这里可以定义页面不同的头部和底部。
- `_config.yml`：配置文件，全局变量在_config.yml中定义，用site.访问。
- `index.html`：页面首页。
