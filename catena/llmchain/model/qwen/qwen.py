import os
from pydantic import Field, model_validator
from typing import Union, Optional
from ..base import Model

from catena_core.alias.builtin import ModelProvider as Provider

class Qwen(Model):
    """ 通义千问模型，使用 openai 作为 SDK """
    
    # 模型提供方
    provider: Provider = Provider.Qwen
    # 模型API密钥
    api_key: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    
        
    def get_client(self):
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = Op
    
    @model_validator(mode="after")
    def validate_model_args(cls, values) -> None:
        model = values.get("model")
        
    
    
class Qwen_DashScope(Qwen):
    """ 通义千问模型，使用 dashscope 作为 SDK """
    
    # 模型提供方
    provider: Provider = Provider.Qwen
    # 模型API密钥
    api_key: Optional[str] = os.getenv("DASHSCOPE_API_KEY")