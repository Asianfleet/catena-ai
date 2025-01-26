from __future__ import annotations
from enum import Enum
from collections import deque
from catenaconf import Catenaconf, KvConfig
from pydantic import BaseModel, Field, ConfigDict
from typing import (
    Any,
    Dict,
    Optional,
    overload
)

from ...error.chainerr import *
from ...settings import RTConfig as RT
from ..alias.builtin import NodeType
from ..callback.node_callback import NodeCallback
        
class NodeCompletion(BaseModel):
    """ 存储每个节点的 operate 函数输出 """
    
    # 节点输出类型
    type: Enum = NodeType.UDFN
    # 主要数据
    main_data: Optional[Any] = None
    # 额外数据池，用键值对存储，这里设置为必选以防止忘记数据传递
    extra_data: RT
    # 回调函数配置
    callback: NodeCallback = NodeCallback()
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def dict(self):
        """ 将 NodeBus 对象转换为字典 """
        _dict = self.__dict__.copy()
        
        # 处理枚举类型的序列化
        from enum import Enum
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict
    
    @property
    def list(self):
        """ 将 NodeBus 对象的非空属性转换为列表 """
        prop_list = []
        for attr, value in self.__dict__.items():
            if value:
                if isinstance(value, NodeCallback):
                    prop_list.append(attr + ": " + str(value.data))
                elif isinstance(value, NodeType):
                    prop_list.append(attr + ": " + str(value.value))
                else:
                    prop_list.append(attr + ": " + str(value))
        return prop_list
    
    @overload
    def update(self, item: NodeCompletion) -> None:
        ...
        
    @overload
    def update(self, **kwargs) -> None:
        ...
        
    def update(self, item: NodeCompletion | None = None, **kwargs) -> None:
        if item is not None:
            if not isinstance(item, NodeCompletion):
                raise TypeError("Item must be of type NodeCompletion")
            self.main_data = item.main_data
            self.extra_data = item.extra_data
        elif kwargs:
            if kwargs.get("main_data"):
                self.main_data = kwargs.get("main_data")
            if kwargs.get("extra_data"):
                self.extra_data = kwargs.get("extra_data")
        else:
            raise ValueError("Must provide either NodeCompletion object or kwargs")
    
class NodeBus(deque[Dict[str, NodeCompletion]]):
    """ 存储每个节点的 operate 函数输出 """
        
    @property        
    def latest(self) -> NodeCompletion:
        """ 获取最新的 NodeCompletion 对象 """
        try:
            return self[-1]["value"]
        except IndexError:
            raise NodeOutputTypeError("NodeCompletion is empty.")
    
    
    @overload
    def add(self, item: NodeCompletion) -> None:
        ...
        
    @overload
    def add(self, **kwargs) -> None:
        ...
    
    def add(self, item: NodeCompletion | None = None, **kwargs) -> None:
        if item is not None:
            if not isinstance(item, NodeCompletion):
                raise TypeError("Item must be of type NodeCompletion")
            super().append({
                "type": item.type.value if isinstance(item.type, Enum) else item.type,
                "value": item
            })
        elif kwargs:
            completion = NodeCompletion(**kwargs)
            super().append({
                "type": completion.type.value if isinstance(completion.type, Enum) else completion.type,
                "value": completion
            })
        else:
            raise ValueError("Must provide either NodeCompletion object or kwargs")
        
    def __call__(self):
        return self.latest
    
