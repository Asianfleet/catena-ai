from abc import abstractmethod
from pydantic import Field, model_validator
from typing import (
    Any, 
    Callable, 
    Dict, 
    Iterator, 
    List, 
    Literal,
    Optional,
    Sequence,
    Union
)

from .metrics import Metrics
from ..response import ModelResponse
from ..message import MessageBus

from ...cli.tools import debug, error
from ...error.chainerr import *
from ...catena_core.node.base import Node, NodeBus
from ...settings import settings, RTConfig
from ...catena_core.callback.node_callback import node_callback
from ...catena_core.alias.builtin import (
    NodeType as Ntype, 
    ModelProvider as Provider
)


class Model(Node):
    """ 大语言模型调用类，用于正式生成文本内容 """

    # 模型名称
    model: str = None
    # 模型提供方
    provider: Provider = None
    # 模型指标
    metrics: Optional[Dict] = Field(default_factory=dict, init=False)
    # openai 风格的工具列表
    tools: Optional[List[Dict]] = None
    # 工具调用方式。
    # 当不存在任何功能时，“none”是默认值。 
    # 如果函数存在，“auto”是默认值。
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # 工具调用序列
    tool_call_sequence: Optional[List[Any]] = None
    # 工具调用次数限制
    tool_call_limit: int = None
    # 模型运行时配置
    rtconfig: Optional[RTConfig] = None
    # 模型 API key
    api_key: Optional[str] = None
    # 模型 client
    client: Any = None
    # 调用代理或工作流的会话 ID。
    session_id: Optional[str] = None
    # 是否在此模型中使用结构化输出。
    structured_outputs: Optional[bool] = None
    # 模型是否支持结构化输出。
    supports_structured_outputs: bool = False
    
    # TODO:这部分还不清楚要不要添加
    # System prompt from the model added to the Agent.
    #system_prompt: Optional[str] = None
    # Instructions from the model added to the Agent.
    #instructions: Optional[List[str]] = None
    
    # 节点 ID
    node_id: Ntype = Field(default=Ntype.MODEL, init=False)
    
    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(include={"model", "node_id", "provider", "metrics"})
        #if self.tools_builtin:
        #    _dict["tools_builtin"] = {k: v.to_dict() for k, v in self.tools_builtin.items()}
        if self.tools:
            _dict["tools"] = self.tools
        if self.tool_call_limit:
            _dict["tool_call_limit"] = self.tool_call_limit

        # 处理枚举类型的序列化
        from enum import Enum
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict
    
    @property
    def completion_args(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @model_validator(mode="after")
    def validate_model_args(cls, values):
        return values
    
    @abstractmethod
    def init_client(self) -> Any:
        """ 获取模型 client """
        pass

    @abstractmethod
    def create_completion(self, messages: List[MessageBus]) -> ModelResponse:
        pass
    
    @abstractmethod
    def create_completion_stream(self, messages: List[MessageBus]) -> Iterator[ModelResponse]:
        pass
    
    @abstractmethod
    async def acreate_completion(self, messages: List[MessageBus]) -> ModelResponse:
        pass
    
    @abstractmethod
    async def acreate_completion_stream(self, messages: List[MessageBus]) -> Any:
        pass
    
    @abstractmethod
    def operate(self, input: Union[NodeBus, Any]) -> NodeBus:
        """ 链式调用语言模型的接口 """
        pass
    
    @node_callback
    def _callback(self, *args, **kwargs) -> Any:
        """ 默认回调函数 """
        pass
    
    #def get_tools_from_builtin(self) -> List[Dict]:
    #    """ 获取模型工具列表 """
    #    pass
    
    def run_function_call(self, *args, **kwargs) -> Iterator[ModelResponse]:
        pass
    
    def deactivate_tool_calls(self) -> None:
        # 通过将未来的工具调用设置为“none”来停用工具调用
        # 当达到函数调用限制时触发。
        self.tool_choice = "none"
        
    def add_images_to_message(message: MessageBus, images: Optional[Sequence[Any]] = None) -> MessageBus:
        """
        将图像添加到模型的消息中。默认情况下，我们使用 OpenAI 图像格式，但其他模型
        可以覆盖此方法以使用不同的图像格式。
        - 参数：
            - message：模型的消息
            - 图像：各种格式的图像序列：
                - str：base64 编码的图像、URL 或文件路径
                - Dict：预先格式化的图像数据
                - 字节：原始图像数据
        - 返回：
            - 以模型期望的格式添加图像的消息内容
        """
        from catena_core.utils.image import process_model_image
        # 如果没有提供图片，则按原样返回消息
        if images is None or len(images) == 0:
            return message

        # 忽略非字符串消息内容
        # 因为我们假设图像/音频已经添加到消息中
        if not isinstance(message.content, str):
            return message

        # 使用文本创建默认消息内容
        message_content_with_image: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]

        # 在消息内容中添加图片
        for image in images:
            try:
                image_data = process_model_image(image)
                if image_data:
                    message_content_with_image.append(image_data)
            except Exception as e:
                error(f"Failed to process image: {str(e)}")
                continue

        # 使用图像更新消息内容
        message.content = message_content_with_image
        return message


if __name__ == "__main__":
    # python -m catena.llmchain.model.base
    Model()