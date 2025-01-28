from typing import Dict, List, Union
from openai import OpenAI

from ....settings import settings


class Role:
    """
    - 用于处理和包装大模型对话提示信息的工具类。功能是将不同角色的对话内容（例如系统、用户、助手等）格式化。
    - 该类包含多种方法，支持文本的预处理、角色标签的赋值以及对话上下文的处理，确保不同角色的输入能够被正确地传递给模型进行响应。
    - 此外还扩展了对图像和文本混合输入的支持，能够将用户提供的图像转换为 base64 编码，并将其与文本信息一起发送给模型。
    """

    @classmethod
    def preprocess(cls, content: str):
        return content.strip().strip("\t")

    @classmethod
    def mask(cls, content: str = None) -> Dict:
        content = content or "你是一位优秀的助手"
        return {"role": "system", "content": cls.preprocess(content)}
    
    @classmethod
    def context(cls, content: str) -> Dict:
        content = "上下文信息：\n" + content
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def system(cls, content: str) -> Dict:
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def assistant(cls, content: str) -> Dict:
        return {"role": "assistant", "content": cls.preprocess(content)}
    
    @classmethod
    def user(cls, content: str) -> Dict:
        return {"role": "user", "content": cls.preprocess(content)}

def init_minimal_client(provider: str = "OpenAI"):
    if provider == "OpenAI":
        return OpenAI()
    else:
        raise ValueError(f"Unknown model {provider}")
    

def minimal_llm_response(
    model: str = "gpt-4o-mini",
    messages: Union[List, str] = None,
    **kwargs
) -> str:
    
    client = init_minimal_client()
    if kwargs:
        response_args = settings.minimal_llm.config.__container__.update(kwargs)
    else:
        response_args = settings.minimal_llm.config.__container__
    
    if isinstance(messages, str):
        messages = [Role.user(messages)]    
        
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **response_args
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # python -m catena_ai.llmchain.model.minimal.llm
    print(minimal_llm_response(
        model="gpt-4o-mini",
        messages=[
            Role.user("你好")
        ]
    ))