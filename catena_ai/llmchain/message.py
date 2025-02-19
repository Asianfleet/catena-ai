from __future__ import annotations
import copy
from time import time
from collections import deque
from pydantic import BaseModel, Field, field_validator
from typing import (
    Any,
    Callable,
    Dict, 
    List, 
    Literal, 
    Optional, 
    overload,
    Sequence, 
    Union
)

from ..settings import settings

class MessageRole:
    """
    - 用于处理和包装大模型对话提示信息的工具类。功能是将不同角色的对话内容（例如系统、用户、助手等）格式化。
    - 该类包含多种方法，支持文本的预处理、角色标签的赋值以及对话上下文的处理，确保不同角色的输入能够被正确地传递给模型进行响应。
    - 此外还扩展了对图像和文本混合输入的支持，能够将用户提供的图像转换为 base64 编码，并将其与文本信息一起发送给模型。
    """

    @classmethod
    def preprocess(
        cls, 
        content: str = None, 
        proc: Callable[[str], str] = None
    ) -> str:
        if proc:
            return proc(content)
        return content.strip().strip("\t")

    @classmethod
    def mask(
        cls, 
        content: str = None, 
        proc: Callable[[str], str] = None
    ) -> Message:
        content = content or "你是一位优秀的助手"
        content = cls.preprocess(content, proc)
        return Message(role="system", content=content)
    
    @classmethod
    def context(
        cls,
        content: str, 
        proc: Callable[[str], str] = None
    ) -> Message:
        content = "上下文信息：\n" + cls.preprocess(content, proc)
        return Message(role="system", content=content)

    @classmethod
    def msg(
        cls, 
        role: Literal["system", "user", "assistant"],
        content: str, 
        proc: Callable[[str], str] = None,
        **kwargs
    ) -> Message:
        content = cls.preprocess(content, proc)
        return Message(role=role, content=content, **kwargs)

    @classmethod
    def system(
        cls, 
        content: str, 
        proc: Callable[[str], str] = None,
        **kwargs
    ) -> Message:
        content = cls.preprocess(content, proc)
        return Message(role="system", content=content, **kwargs)

    @classmethod
    def assistant(
        cls, 
        content: str, 
        proc: Callable[[str], str] = None
    ) -> Message:
        content = cls.preprocess(content, proc)
        return Message(role="assistant", content=content)
    
    @classmethod
    def user(
        cls, 
        content: str, 
        proc: Callable[[str], str] = None,
        **kwargs
    ) -> Message:
        content = cls.preprocess(content, proc)
        return Message(role="user", content=content, **kwargs)
    
class MessageContext(BaseModel):
    pass

class Message(BaseModel):
    """ 输入大模型 SDK 的消息封装 """
    
    # 角色
    role: Literal["system", "user", "assistant", "tool"]
    # 内容
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    # 工具调用
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # 内置的工具调用
    tool_call_builtin: Optional[List[Any]] = None
    # 工具调用 ID
    tool_call_id: Optional[str] = None
    # 是否在工具调用后停止对话
    stop_after_tool_call: Optional[bool] = False
    
    # An optional name for the participant. Provides the model information 
    # to differentiate between participants of the same role.
    name: Optional[str] = None
    
    # 其他模态信息
    audio: Optional[Any] = None
    images: Optional[Union[str,List]] = None
    videos: Optional[Sequence[Any]] = None
    
    # 上下文信息，主要与 RAG 结合 
    context: Optional[MessageContext] = None
    
    # 运行时模型参数，优先级最高
    runtime_args: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # 模型单次调用性能信息
    metrics: Optional[Dict] = Field(default_factory=dict)
    
    # 创建时间
    created_at: int = Field(default_factory=lambda: int(time()))

    @field_validator("role")
    def role_validator(cls, role: str, info) -> str:
        """验证assistant role的多媒体数据"""
        if role == "assistant":
            data = info.data
            if data.get("images") is not None:
                raise ValueError("Assistant message cannot contain images.")
            if data.get("videos") is not None:
                raise ValueError("Assistant message cannot contain videos.")
            if data.get("audio") is not None:
                raise ValueError("Assistant message cannot contain audio.")
        return role

    @property
    def model_message(self) -> Dict[str, Any]:
        return self.to_model_message()

    def to_model_message(self) -> Dict[str, Any]:
        """ 将 Message 对象转换为模型消息列表 """
    
        """ if self.role == "system":
            with open("system.txt", "w") as attr:
                attr.write(str(self.model_dump_json()))
        else:
            with open("user.txt", "w") as attr:
                attr.write(str(self.model_dump_json())) """
        
        _message = self.model_dump(
            exclude_none=True,
            include={"role", "content", "audio", "name", "tool_call_id", "tool_calls"},
        )
        if self.content is None:
            _message["content"] = None
        
        if self.images:
            # 处理带图片的消息
            from ..catena_core.utils.image import to_base64
            images = self.images if isinstance(self.images, List) else [self.images]
            _message["content"] = []
            for img in images:
                image_base64 = to_base64(img)
                _message["content"].append({
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                })
            _message["content"].append({"type": "text", "text": self.content})
            return _message

        return _message

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            import json

            return json.dumps(self.content)
        return ""

    def printf(self, **kwargs):
        """ 打印 Message 对象的信息 """
        from ..cli.tools import info, info_condition
        kwargs.update({"pre": False})
        with info_condition(settings.visualize.message_metrics, **kwargs):
            info("**************** MESSAGE OBJECT ****************")
            info(f"* Role:                   {self.role}")
            info(f"* Content:                {self.content}")
            if self.tool_calls is not None:
                info(f"* Tool calls:             {self.tool_calls}")
            if self.tool_call_builtin is not None:
                info(f"* Builtin tool calls:     {self.tool_call_builtin}")
            if self.tool_call_id is not None:
                info(f"* Tool call ID:           {self.tool_call_id}")
            if self.audio is not None:
                info(f"* Audio:                  {self.audio}")
            if self.images is not None:
                info(f"* Images:                 {self.images}")
            if self.videos is not None:
                info(f"* Videos:                 {self.videos}")
            if self.context is not None:
                info(f"* Context:                {self.context}")
            if self.metrics is not None:
                info(f"* Metrics:                {self.metrics}")
            info(f"* Created at:             {self.created_at}")
            info("**************** MESSAGE OBJECT ****************")

class MessageBus(deque[Message]):
    """ 存储每个节点的 operate 函数输出 """
        
    @property        
    def latest(self) -> Message:
        """ 获取最新的 Message 对象 """
        if not self:
            raise IndexError("Message is empty.")
        return self[-1]
    
    @property
    def model_message(self) -> List[Dict[str, Any]]:
        """ 将 Message 对象转换为模型消息列表 """
        return [message.to_model_message() for message in self]
    
    @overload
    def add(self, item: Message) -> None:
        ...
        
    @overload
    def add(self, **kwargs) -> None:
        ...
    
    def add(self, item: Message | None = None, **kwargs) -> None:
        if item is not None:
            if not isinstance(item, Message):
                raise TypeError("Item must be of type Message")
            super().append(item)
        elif kwargs:
            message = Message(**kwargs)
            super().append(message)
        else:
            raise ValueError("Must provide either Message object or kwargs")
        
    def deepcopy(self) -> MessageBus:
        """ 深度复制 MessageBus 对象 """
        bus = MessageBus()
        for message in self:
            bus.add(copy.deepcopy(message))
        return bus
        
    def extend(self, item: Union[Message, MessageBus]) -> None:
        """往队列右侧添加一个Message或MessageBus对象
        
        Args:
            item: Message或MessageBus对象
                
        Raises:
            TypeError: 当输入类型不是Message或MessageBus时抛出
        """
        if isinstance(item, Message):
            super().append(item)
        elif isinstance(item, MessageBus):
            for msg in item:
                super().append(msg)
        else:
            raise TypeError("Item must be of type Message or MessageBus")

    def __call__(self):
        return self.latest

if __name__ == "__main__":
    # python -m catena_ai.llmchain.message
    message = Message(
        role="user",
        content="nihao",
        images=[
            "https://www.baidu.com/img/bd_logo1.png",
            "https://www.baidu.com/img/bd_logo1.png"
        ],
        tool_calls=[{'id': 'call_EcKtTB07sQGadJgxMWVIdhnG', 'function': {'arguments': '{"region":"齐河"}', 'name': 'sample_func'}, 'type': 'function'}]
    )
    print(message.to_model_message())
    
