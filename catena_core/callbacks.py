from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Callback:
    """ 回调函数的数据结构 """
    source: Optional[str] = field(default=None, metadata={"description": "回调函数请求发起源"})
    target: Optional[str] = field(default=None, metadata={"description": "回调函数请求接收目标"})
    name: Optional[str] = field(default=None, metadata={"description": "回调函数名称"})
    main_input: Optional[Any] = field(default=None, metadata={"description": "回调函数接收的主要参数"})
    args: Optional[tuple] = field(default_factory=tuple, metadata={"description": "回调函数接收的位置参数"})
    kwargs: Optional[dict] = field(default_factory=dict, metadata={"description": "回调函数接收的关键字参数"})
    
    @property
    def data(self):
        return self.to_dict(exclude_none=False)
    
    @property
    def list(self):
        """ 将 Callback 对象转换为列表 """
        return [f"{attr}: {value}" for attr, value in self.to_dict().items()]

    def to_dict(self, exclude_none=True):
        """ 将数据类转换为字典，排除值为 None 的属性 """
        if exclude_none:
            return {k: v for k, v in self.__dict__.items() if v is not None}
        else:
            return self.__dict__
    
    def __bool__(self):
        return all(attr is not None for attr in self.__dict__.values())