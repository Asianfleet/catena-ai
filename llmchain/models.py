from __future__ import annotations
import os
import qianfan
from openai import OpenAI
from typing import Union, Callable, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..catena_core.error.chainerr import *
from ..catena_core.utils import PromptRole
from ..catena_core.nodes import Node, NodeCompletion
from ..catena_core.alias import BuiltInType, BuiltInID
from ..catena_core.settings import settings, RTConfig
from ..catenasmith.visualize.cli_visualize import cli_visualize

def init_model(
    platform: str = "Kimi", model:str=None, max_tokens:int=2048, temperature:float=0.5, output_type:str="text"
) -> Union[GPT, QianFan, Claude]:

    if platform == "GPT":
        print("[Init Model] 初始化 GPT 系列...")
        return GPT()
    elif platform == "QianFan":
        print("[Init Model] 初始化 QianFan 系列...")
        return QianFan(model, max_tokens, temperature, output_type)
    elif platform == "Claude":
        print("[Init Model] 初始化 Claude 系列...")
        return Claude()
    else:
        raise ValueError(f"Unknown model {platform}")

class ModelArgs(BaseModel):
    """ 模型参数 """
    model: str = Field(default=None, title="Shared", description="模型名称")
    max_tokens: int = Field(default=None, gt=0, title="Shared", description="最大 token 数")
    top_p: float = Field(default=None, gt=0.0,lt=1.0,title="Shared", description="核采样参数")
    temperature: float = Field(default=None, title="Shared", description="温度")
    stream: bool = Field(default=None, title="Shared", description="是否开启流式输出")
    stream_options: dict = Field(default=None, title="Shared", description="流式输出的配置")
    tools: List[dict] = Field(default=None, title="Shared", description="function call 工具列表")
    web_search: bool = Field(default=None, title="Special", description="是否开启在线搜索")
    enable_search: bool = Field(default=None, title="Special", description="是否开启在线搜索")
    presence_penalty: float = Field(default=None, ge=-2.0,le=2.0,title="Special", description="存在惩罚")

    @field_validator('enable_search', 'web_search')
    def check_search_fields(cls, value, values):
        if value is not None and values.get('web_search') is not None:
            raise ValueError("enable_search 和 web_search 不能同时为非空")
        return value
    
MAPPING:dict = {
        "moonshot": {
            "url": "https://api.moonshot.cn/v1", 
            "api_key": os.getenv("MOONSHOT_KEY")
        },
        "qwen": {
            "url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
            "api_key": os.getenv("DS_KEY")
        },
        "ernie": {
            "url": "https://qianfan.baidubce.com/v2", 
            "api_key": os.getenv("QF_BT_TOKEN")
        },  
        # TODO:妈的，百度的傻逼认证机制太他妈麻烦了，先要通过 API 获取 BearerToken， 
        # 但是这个 API 本身还需要两个 Key 来进行验证才能调用，妈个逼的太他妈傻逼了艹，以后有时间再搞吧
        # 其他与 OpenAI 兼容的模型都他妈不需要这么麻烦，一个 Key 就行了，就他妈百度是个异类
    }    

class OpenAIC(OpenAI, Node):
    """ 支持 OpenAI SDK 调用的模型（OpenAI Compatible） """
    
    id = BuiltInID.LM
    
    def __init__(
        self, 
        model:str,
        tpa:bool=False, 
        max_tokens:int=None,
        top_p:float=None,
        temperature:float=None,
        **kwargs
    ):
        Node.__init__(self, "bold purple", f"OpenAIC({model})")
        self._model:str = model                                             # 模型名称
        self._use_thirdparty_aggregator:bool = tpa                          # 是否使用第三方聚合平台
        self._max_tokens:int = max_tokens or settings.llm.max_tokens        # 最大 token 数
        self._top_p:float = top_p or settings.llm.top_p                     # 核采样参数
        self._temperature:float = temperature or settings.llm.temperature   # 温度
        self._model_args = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        } | kwargs
        
        self.__post_init__()
    
    def __post_init__(self):
        """ 初始化模型服务 """
        
        model_type = self._model.split("-")[0]
        try:
            if not self._use_thirdparty_aggregator: # 未使用第三方聚合平台（使用模型提供方的 API 服务）
                
                if model_type not in MAPPING.keys():  # 模型类型不存在
                    raise LLModelInitError(f"Unknown model type {self._model}")
                if MAPPING[model_type]["api_key"] is None:    # API Key 为空
                    raise LLModelInitError(f"api_key is empty")
                url = MAPPING[model_type]["url"]
                api_key = MAPPING[model_type]["api_key"]
            else:   # 使用第三方聚合平台（使用第三方聚合平台的 API 服务）
                self._url:str = settings.llm.thirdparty_url          # 第三方聚合平台 URL（非必须）
                self._api_key:str = settings.llm.thirdparty_api_key  # 第三方聚合平台 API Key（非必须）
                if not self._url or not self._api_key:  # URL 或 API Key 为空
                    raise LLModelInitError("url or api_key is empty")
                url = self._url
                api_key = self._api_key
        except LLModelInitError as e:
            e.trace()
        # 初始化 OpenAI SDK
        self.client = OpenAI(api_key=api_key, base_url=url)
        
    def query(self, message: List[Dict], **kwargs) -> Dict:
        """ 通过 OpenAI SDK 调用语言模型 """
        
        completion = self.client.chat.completions.create(
            model = self._model,
            messages = message,
            max_tokens = self._max_tokens,
            temperature = self._temperature,
        )
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
        else:
            args_ = self._model_args
        self._cache = {"input": input, "config": config}
        
        # 使用 model_validate 方法来验证字典
        try:
            valid_args = ModelArgs.model_validate(args_)
            args_ = valid_args.model_dump(exclude_none=True)
            args_.update({"messages": input})
        
        except ValidationError as e:
            print("Validation error:", e)
        # 调用 create 方法进行处理，并将结果赋值给 llmreply
    
        completion = self.client.chat.completions.create(**args_)
        reply = completion.choices[0].message
        llmreply = PromptRole.assistant(reply.content)

        # 返回处理结果
        return NodeCompletion(BuiltInType.LLMOP, llmreply, args, kwargs, config)
    
    @Node.callback("regen")
    def regenerate(self, input: Callable[[Any], int | str]) -> NodeCompletion:
        while True:
            regen = self.operate(**self._cache)
            checked_result = input(regen)
            if checked_result != -1:
                continue
            else:
                break
        return regen
        
class Qwen(OpenAI, Node):
    """ Qwen 系列大语言模型调用类，用于正式生成文本内容 """

    id = BuiltInID.LM

    def __init__(self, model: str="qwen-turbo", **kwargs) -> None:
        Node.__init__(self, "bold purple")
        self._model = model
        self._temperature = None
        self._client:OpenAI = OpenAI(api_key=os.getenv("DS_KEY"), base_url=MAPPING["qwen"]["url"])
        
    def query(self, message: List[Dict], *args, **kwargs) -> Dict:
        
        with open("data.json", "w") as f:
            import json
            json.dump({"msg": message}, f)
        
        completion = self._client.chat.completions.create(
            model = kwargs.get("model", self._model), 
            messages = message,
            max_tokens = kwargs.get("max_tokens", 2048), 
            temperature = kwargs.get("temperature", 0.3),
        )

        reply = completion.choices[0].message

        result = reply.content 

        return PromptRole.assistant(result) 
    
    #visualize
    def operate(
        self, input: dict, config: RTConfig = None, *args, **kwargs
    ) -> dict:
        self._cache = input
        config = config or RTConfig()
        # 调用 query 方法进行处理，并将结果赋值给 llmreply
        llmreply = self.query(input, *args, config.unwrap, **kwargs)
        # 返回处理结果
        return NodeCompletion(BuiltInType.LLMOP, llmreply, args, kwargs, config=config)
    
    @Node.callback("regen")
    def regenerate(self, *args, **kwargs) -> NodeCompletion:
        self._signature = self.get_name() + "(CallBack: regenerate)"
        regen:NodeCompletion = self.operate(self._cache, *args, **kwargs)
        self._signature = self.get_name()
        return regen

class QianFan(Node):
    """ 百度千帆平台调用类，用于正式生成文本内容 """
    pass

class GPT(OpenAI, Node):
    pass

class Claude(OpenAI, Node):
    pass

if __name__ == "__main__":
    # python -m src.modules.agent.llmchain.models
    print("Hello World")
    
    data = {
        "top_p": 0.8
    }
    
    try:
        # 使用 parse_obj 方法来验证字典
        user = ModelArgs.model_validate(data)
        print("Validation successful:", user.model_dump(exclude_none=True))
    except ValidationError as e:
        print("Validation error:", e)