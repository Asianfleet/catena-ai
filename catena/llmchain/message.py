from time import time
from collections import deque
from pydantic import BaseModel, Field
from typing import (
    Any,
    Dict, 
    List, 
    Literal, 
    Optional, 
    overload,
    Sequence, 
    Union
)

from ..settings import settings
from ..catena_core.paths import System

class MessageRole:
    """
    - 用于处理和包装大模型对话提示信息的工具类。功能是将不同角色的对话内容（例如系统、用户、助手等）格式化。
    - 该类包含多种方法，支持文本的预处理、角色标签的赋值以及对话上下文的处理，确保不同角色的输入能够被正确地传递给模型进行响应。
    - 此外还扩展了对图像和文本混合输入的支持，能够将用户提供的图像转换为 base64 编码，并将其与文本信息一起发送给模型。
    """

    @classmethod
    def preprocess(cls, content: str = None):
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
    
    @classmethod
    def user_vision(cls, text: str, image:List) -> Dict:
        from catena_core.utils.image import to_base64, concat_images
        if settings.prompt.image_concat_direction:
            direction = settings.prompt.image_concat_direction
            conact_savepath = System.DEBUG_DATA_PATH.val("image") + "/concated.png"
            img_concated = concat_images(image, direction, conact_savepath)
            image_base64 = to_base64(img_concated)
            msg = {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                },
                {"type": "text", "text": text}
            ]}
        else:
            msg = {"role": "user", "content": []}
            for img in image:
                image_base64 = to_base64(img)
                msg["content"].append({
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                })
            msg["content"].append({"type": "text", "text": text})
        return msg

class MessageContext(BaseModel):
    pass

class Message(BaseModel):
    """ 输入大模型 SDK 的消息封装 """
    
    # 角色
    role: Literal["system", "user", "assistant", "tool"]
    # 内容
    content: Union[str, List[Dict[str, Any]]]
    # 工具调用
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # 内置的工具调用
    tool_call_builtin: Optional[List[Any]] = None
    # 工具调用 ID
    tool_call_id: Optional[str] = None
    
    # 其他模态信息
    audio: Optional[Any] = None
    images: Optional[Sequence[Any]] = None
    videos: Optional[Sequence[Any]] = None
    
    # 上下文信息，主要与 RAG 结合 
    context: Optional[MessageContext] = None
    
    # 模型单次调用性能信息
    metrics: Optional[Any] = None
    
    # 创建时间
    created_at: int = Field(default_factory=lambda: int(time()))
    
    def to_dict(self) -> Dict[str, Any]:
        """ 将 MessageBus 对象转换为字典，用于发送给模型 """
        _dict = self.model_dump(
            exclude_none=True,
            include={"role", "content"},
        )
        # 手动添加内容字段，即使它是 None
        if self.content is None:
            _dict["content"] = None
        from enum import Enum
        # 处理枚举类型的序列化
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            import json

            return json.dumps(self.content)
        return ""

    def printf(self):
        """ 打印 Message 对象的信息 """
        from ..catenasmith.cli_tools import info
        info("**************** MESSAGE START ****************")
        info(f"* Role:                       {self.role}")
        info(f"* Content:                    {self.content}")
        if self.tool_calls is not None:
            info(f"* Tool calls:                 {self.tool_calls}")
        if self.tool_call_builtin is not None:
            info(f"* Builtin tool calls:         {self.tool_call_builtin}")
        if self.tool_call_id is not None:
            info(f"* Tool call ID:               {self.tool_call_id}")
        if self.audio is not None:
            info(f"* Audio:                      {self.audio}")
        if self.images is not None:
            info(f"* Images:                     {self.images}")
        if self.videos is not None:
            info(f"* Videos:                     {self.videos}")
        if self.context is not None:
            info(f"* Context:                    {self.context}")
        if self.metrics is not None:
            info(f"* Metrics:                    {self.metrics}")
        info(f"* Created at:                 {self.created_at}")
        info("**************** MESSAGE END ******************")

class MessageBus(deque[Message]):
    """ 存储每个节点的 operate 函数输出 """
        
    @property        
    def latest(self) -> Message:
        """ 获取最新的 Message 对象 """
        if not self:
            raise IndexError("Message is empty.")
        return self[-1]
    
    @overload
    def update(self, item: Message) -> None:
        ...
        
    @overload
    def update(self, **kwargs) -> None:
        ...
    
    def update(self, item: Message | None = None, **kwargs) -> None:
        if item is not None:
            if not isinstance(item, Message):
                raise TypeError("Item must be of type Message")
            super().append(item)
        elif kwargs:
            message = Message(**kwargs)
            super().append(message)
        else:
            raise ValueError("Must provide either Message object or kwargs")
        
    def __call__(self):
        return self.latest

if __name__ == "__main__":
    # python -m catena.llmchain.message
    message = Message(
        role="user",
        content="nihao"
    )
    
    bus = MessageBus()
    bus.update(message)
    bus.update(role="assistant", content="nihao")
    print(bus)