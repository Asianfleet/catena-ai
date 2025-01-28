import os
from enum import Enum
from packaging import version
from pydantic import BaseModel, ConfigDict
from pydantic import Field, model_validator
from typing import (
    Any, 
    Dict, 
    Iterator, 
    List, 
    Optional, 
    Union
)

from .oai import OpenAIOrigin
from ..base import Model
from ..metrics import Metrics
from ...response import ModelResponse
from ...message import Message, MessageBus
from ....error.modelerr import ModelError
from ....cli.tools import info, warning
from ....catena_core.alias.builtin import (
    NodeType as Ntype, 
    ModelProvider as Provider
)
from ....catena_core.callback import NodeCallback
from ....catena_core.node.completion import NodeCompletion

class OpenAI_C(OpenAIOrigin):
    
    # 模型名称
    model: str = "gpt-4o-mini"
    # 模型提供商
    provider: Enum = Field(default=Provider.OpenAIC, init=False)
    # 模型 URL
    base_url: Optional[str] = None
    # 模型 API Key
    api_key: Optional[str] = None
    
    # 内部参数。不用于 API 请求
    # 是否使用结构化输出与此模型。
    structured_outputs: bool = False
    # 模型是否支持结构化输出。
    supports_structured_outputs: bool = True
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )