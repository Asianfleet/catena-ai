from dataclasses import dataclass, field
from typing import Optional, Any

from catenasmith.cli_tools import info, debug

from catena_core.error.chainerr import NodeCallbackNotFoundError

@dataclass
class Callback:
    """ 节点回调函数的数据结构 """
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
    
class NodeCallbackRegister:
    
    _callbacks = {}
    
    @classmethod
    def node_callback(cls, method_name: str):
        """ 装饰器，用于注册回调方法 """
        def decorator(func):
            cls._callbacks[method_name] = func
            return func
        return decorator

    @classmethod
    def run_callback(
        cls, 
        instance: Any, 
        method_name: str, 
        main_input: Any,
        *args, 
        **kwargs
    ):  
        """
        执行指定名称的回调方法。
        
        该类方法用于根据提供的方法名称，从类的回调方法字典中查找并执行相应的回调方法。
        如果找到了对应的回调方法，则将其绑定到提供的实例上，并使用提供的参数调用该方法。
        
        参数:
        - instance: Any 类型，表示要执行回调方法的实例。
        - method_name: str 类型，表示要执行的回调方法的名称。
        - main_input: Any 类型，表示回调方法的主要输入参数。
        - *args: 传递给回调方法的额外位置参数。
        - **kwargs: 传递给回调方法的额外关键字参数。
        
        返回:
        - 成功执行回调方法的结果，如果未找到回调方法则返回 None。
        """
        debug("[run_callback] callbacks:", cls._callbacks)
        debug("[run_callback] running callback...")
        debug("[method_name] name:", method_name)
        
        try:
            # 检查方法名称是否在回调方法字典中
            if method_name in cls._callbacks.keys():
                # 获取回调方法
                method = cls._callbacks[method_name]
                # 确保方法是绑定到实例上的
                bound_method = method.__get__(instance, instance.__class__)
                # 执行回调方法并返回结果
                result = bound_method(main_input, *args, **kwargs)
                return result
            else:
                raise NodeCallbackNotFoundError()
        except NodeCallbackNotFoundError:
            return None
        except Exception as e:
            raise e