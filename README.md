# 概述

本示例使用Ollama在本机部署一个大语言模型，无需使用相关API，借助Langchain和Chain构建一个AI搜索问答系统。在笔记本电脑上即可运行。

![示例](assets/demo.png)

# 环境准备

- 操作系统：macOS、Linux、Windows系统均可，本示例在macOS环境下进行，未对Linux、Windows进行测试
- Ollama：从[Ollama官网](https://ollama.ai/)下载并安装Ollama即可
- Python 3.11.4环境可以正常使用，其他版本未做测试，理论上相关Python包可以安装都可以使用

# 快速开始

## 准备模型

在命令行中，执行如下命令，获取LLM和Embedding模型：

```bash
ollama pull qwen:7b
ollama pull znbang/bge:large-zh-v1.5-q8_0
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动

```bash
sh start.sh
```
