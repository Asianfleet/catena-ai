from __future__ import annotations
import os
import qianfan
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Literal,
    Optional,
    Union
)

from .metrics import Metrics

from catenasmith.cli_tools import debug
from catenasmith.cli_tools import Style as sty
from catenasmith.visualize.cli_visualize import cli_visualize

from catena_core.error.chainerr import *
from catena_core.utils.utils import PromptRole
from catena_core.tools.base import BaseTool
from catena_core.nodes import Node, NodeCompletion
from catena_core.settings import settings, RTConfig
from catena_core.alias.builtin import (
    NodeCompletionType, 
    NodeType, 
    ModelProvider as MP
)
from catena_core.callback.node_callbacks import (
    Callback, 
    NodeCallbackRegister as Register
)

class Model(BaseModel, Node):
    """ Qwen 系列大语言模型调用类，用于正式生成文本内容 """

    # 节点 ID
    _id: Enum = NodeType.LM

    # 模型名称
    model: str = None
    # 模型提供方
    provider: Enum = None
    # 模型指标
    metrics: Optional[Metrics] = None
    # catena 内部定义的模型工具
    tools: Optional[List[BaseTool]] = None
    # openai 风格的工具列表
    tools_openai: Optional[List[Dict]] = None
    # 工具调用次数限制
    tool_call_limit: int = None
    # 模型运行时配置
    model_config: Optional[RTConfig] = None
    # 模型 client
    model_client: Any = None
    
    @property
    def id(self):
        return self._id

    def __init__(
        self, 
        model: str="qwen-turbo", 
        *,
        top_p: float = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> None:
        Node.__init__(self, "bold purple", f"Qwen({model})")
        self._model: str = model                                             # 模型名称
        self._max_tokens: int = max_tokens or settings.llm.max_tokens        # 最大 token 数
        self._top_p: float = top_p or settings.llm.top_p                     # 核采样参数
        self._temperature: float = temperature or settings.llm.temperature   # 温度
        self._model_args = {
            **{
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "top_p": self._top_p,
            }, 
            **kwargs
        }
        self._client: OpenAI = OpenAI(
            api_key=os.getenv("DS_KEY"), 
            base_url=MAPPING["qwen"]["url"]
        )
        
    def query(self, message: List[Dict], *args, **kwargs) -> Dict:
                
        kwargs.update({"messages": message})
        completion = self._client.chat.completions.create(**kwargs)
        reply = completion.choices[0].message
        result = reply.content 

        return PromptRole.assistant(result) 
    
    @cli_visualize
    def operate(
        self, input: list, config: RTConfig = None, *args, **kwargs
    ) -> NodeCompletion:
        """ 调用语言模型 """
        config = config or RTConfig()   # 处理运行配置信息
        
        if config().get("meta"):
            config._merge(self._model_args, "meta.model_args")  # 合并模型参数
            args_ = config("meta.model_args")
            debug("merged_args_:", args_)
        else:
            args_ = self._model_args
            debug("args_:", args_)
        self._cache = {"input": input, "config": config}
        
        # 使用 model_validate 方法来验证字典
        try:
            valid_args = ModelArgs.model_validate(args_)
            args_ = valid_args.model_dump(exclude_none=True)
            args_.update({"messages": input})
        
        except ValidationError as e:
            debug("Validation error:", e)
        # 调用 create 方法进行处理，并将结果赋值给 llmreply
    
        completion = self._client.chat.completions.create(**args_)
        reply = completion.choices[0].message
        llmreply = PromptRole.assistant(reply.content)

        # 返回处理结果
        debug("[models] related_chain_ids:", self.related_chain_ids)
        if NodeType.MOM in self.related_chain_ids:
            debug("[models] update memory")
            callback = Callback(
                source=NodeType.LM,
                target=NodeType.MOM,
                name="update_memory",
                main_input=llmreply,
            )
        else:
            callback = None
        return NodeCompletion(
            NodeCompletionType.LLMOP, 
            main=llmreply, 
            args=args, 
            kwargs=kwargs, 
            config=config,
            callback=callback
        )