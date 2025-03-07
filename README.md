<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<p align="center">
    <img src="assets/catena.jpg" alt="catenaconf logo" width=200/>
</p>
<h1 align="center">Catena</h1>

## 简介

Catena 是一个大模型智能体搭建框架，提供模型基本调用、工具调用、Agent 搭建等功能，并支持用户自定义。

## 安装及配置

```shell
pip install catena
```

安装完成后，还需向系统中添加用于存储大模型 api_key 的环境变量。目前支持以下提供方：

- OpenAI: `OPENAI_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`
- DashScope: `DASHSCOPE_API_KEY`
  
## 使用方法

### 模型基本调用

```python
from catena.llmchain.model.oai import OpenAIOrigin

gpt = OpenAIOrigin(
    model="gpt-4o-mini"
)


```
