from __future__ import annotations
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional, 
    Union
)

from ...settings import (
    settings, 
    RTConfig as RT,
    debug
)
from ...error.chainerr import *
from ..alias.builtin import NodeType
from ..callback.node_callback import (
    NodeCallback, 
    NodeCallbackRegister as Register,
    node_callback
)

class NodeMeta(BaseModel.__class__):
    """ Node 的元类，用于实现类之间（不是类的实例）的运算符重载和链式调用 """
    def __rshift__(cls, other):
        debug("[NodeMeta.__rshift__] type of cls:", type(cls))
        debug("[NodeMeta.__rshift__] type of other:", type(other))
        return cls() >> other
    
    def __rrshift__(cls, other):
        debug("[NodeMeta.__rrshift__] type of cls:", type(cls))
        debug("[NodeMeta.__rrshift__] type of other:", type(other))
        return other >> cls()

@dataclass
class NodeBus:
    """ 存储每个节点的 operate 函数输出 """
    
    type: Enum   # 节点输出类型
    main: Any = "None"  # 主输出
    args: tuple = ()    # 位置参数
    kwargs: dict = field(default_factory=dict)  # 关键字参数
    config: RT = RT()   # 运行时配置
    callback: NodeCallback = NodeCallback() # 回调函数配置
    
    @property
    def dict(self):
        """ 将 NodeBus 对象转换为字典 """
        return self.__dict__
        
    @property
    def list(self):
        """ 将 NodeBus 对象的非空属性转换为列表 """
        prop_list = []
        for attr, value in self.__dict__.items():
            if value:
                if hasattr(value, "data"):
                    prop_list.append(attr + ": " + str(value.data))
                elif isinstance(value, Enum):
                    prop_list.append(attr + ": " + str(value.value))
                else:
                    prop_list.append(attr + ": " + str(value))
        return prop_list

class Node(BaseModel, metaclass=NodeMeta):
    """
    链式调用的核心基类（抽象基类），是链中每个节点的基本构建块。任何类可通过继承该类并实现内部接口成为一个节点。
    该基类提供了调试可视化、回调注册和执行等功能：
     - 支持通过 visualize 装饰器对任务执行过程进行调试和实时可视化，允许开发者在任务执行时查看输入输出的变化。
     - 支持通过链式调用 (__or__, __rshift__ 等运算符) 结合 NodeChain 将多个任务组合成一个执行流，形成任务链。
     - 每个 Node 对象都可以执行具体的任务逻辑，通过实现 operate 方法将输入数据传递并返回处理结果。
     - 通过回调机制，用户可以在任务执行过程中进行动态的操作与数据交互。
    """

    # 不能在外部更改的属性
    # 节点ID
    node_id: Enum = Field(default=NodeType.UDFN, init=False)
    # 节点样式
    style: Optional[Union[str, Enum]] = None                               
    # 节点签名
    signature: Optional[str] = None
    # 可在外部更改的属性
    # 前置节点
    former_node: str = None             
    # 后置节点

    
    def __post_init__(self):
        # 关联的链的id
        self.related_chain_ids: List[Enum] = []             
        # 回调函数注册表
        self.callback_register: Dict[str, callable] = {} 
    
    @property
    def node_id(self):
        return self.node_id
    
    @property
    def input_type(self):
        return self.input_type
    
    @property
    def type(self):
        return self.__class__.__name__
    
    @property
    def style(self):
        if isinstance(self.style, Enum):
            return self.style.value
        else:
            return self.style
    
    @property
    def signature(self):
        if self.signature is None:
            self.signature = self.type
        return self.signature
    
    @property
    def callback_register(self):
        return self.callback_register
    
    # TODO: 将__or__和__ror__的重载功能进行替换 
    def __or__(
        self,
        other: Union[
            Callable[[Any], Any],
            Dict[str, Any],
            type,
            Node
        ]
    ) -> NodeChain:
        return NodeChain(self, encapsulate(other))
       
    def __ror__(self, other: Any) -> NodeChain:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(encapsulate(other), self)
     
    """ 
    ############################# 流式调用的核心原理：运算符重载 #################################
     如果两个对象要使用运算符相连，则这两个类必须实现相应的方法。例如，a | b，相当于a.__or__(b)。
     如果 a 中未实现 __or__，则 Python 解释器会尝试调用 __ror__ ，即b.__ror__(a)。
     该类通过对运算符 >> （对应 __rshift__ 和 __lshift__ ）的重载以及 NodeChain 来实现链式调用：
     - NodeChain 的实例化接受两个 Node 对象，并按顺序存储在类内部的列表中。
     - 对于 a >> b，首先调用 a.__rshift__(b)，将实例 a, b 先后存储在 NodeChain 内部，并返回该实例。
     - 若 a 未实现 __rshift__ （未继承 Node 基类），则将尝试 b.__rrshift__(a) ；
       此时由于 a 不是 Node 的实例，故会先调用 encapsulate 方法将 a 转换为 Node 对象。
     - 最后的效果为：a >> b 等价于 NodeChain(a, b) 或者 NodeChain(encapsulate(a), b)。
     - 当有三个级以上的类时（例如 a >> b >> c），a >> b 会先返回 NodeChain 实例，然后该实例会
       调用 NodeChain.__rshift__(c)，将 c 添加到内部的列表中，并返回该实例。
     - 链式调用的好处是可以通过 >> 运算符将多个任务组合成一个链式调用，简化代码。 
    ############################################################################################
    """
    
    def __rshift__(
        self,
        other: Union[
            Callable[[Any], Any],
            Dict[str, Any],
            type,
            Node
        ]
    ) -> NodeChain:
     
        return NodeChain(self, encapsulate(other))
       
    def __rrshift__(self, other: Any) -> NodeChain:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(encapsulate(other), self)
    
    def __call__(self, other):
        return self.__rshift__(other)

    @node_callback("default")
    def _callback(self, main_input: Any, *args, **kwargs) -> Any:
        pass

    def operate( 
        self, input: Any, config: Optional[RT] = None, *args, **kwargs
    ) -> NodeBus:
        """  """
        return NotImplementedError

class NodeChain(Node):
    """
    是 Node 的具体实现，主要用于将多个 Node 任务串联起来，形成一个按顺序执行的任务链：
     - 每个任务的输出会自动作为下一个任务的输入，实现任务的顺序执行。
     - 提供了灵活的链式调用能力，可以将多个 Node 任务组合在一起执行，形成复杂的任务流水线。
     - 在执行过程中，NodeChain 会确保每个任务按顺序执行，并处理回调机制与配置。
     - 支持链式操作符，允许开发者方便地将多个任务组合在一起。
     - operate 方法会依次执行链中的所有任务，并最终返回最后一个任务的输出结果。
    """
    
    first: Node
    second: Node
    
    def __pose_init__(self):
   
        self.chain = [self.first, self.second]
        self.compiled = False
        
    def __call__(self, node: Any):
        self.chain.append(encapsulate(node))
        return self

    #TODO:将__or__和__ror__的重载功能进行替换 
    def __or__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
        ],
    ) -> "NodeChain":
        return self.__call__(other)

    def __rshift__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
            type
        ],
    ) -> "NodeChain":
        return self.__call__(other)

  

    def operate(
        self, input: Optional[Any] = None, config: RT = None, *args, **kwargs
    ):
    
        return ""
    
class SNode(Node):
    # 不能在外部更改的属性
    # 节点ID
    node_id: Enum = Field(default=NodeType.MODEL, init=False)
    # 节点样式
    style: Union[str, Enum] = None
    
    def operate( 
        self, input: Any, config: Optional[RT] = None, *args, **kwargs
    ) -> NodeBus:
        
        return ""
    
    
def encapsulate(obj: Any) -> Node:
    """ 辅助函数用于将其他对象转换为 Node 对象 """
    # 下面使用 SimpleNode 类将非 Node 类的对象封装成 Node 类
    if isinstance(obj, Node):   # 其 operate 函数的输出类型为 WRAPPED，代表是封装节点
        # 如果对象已经是 Node 对象（或其子类）的实例，直接返回
        debug("[encapsulate] obj is a Node instance")
        result = obj
    elif type(obj) is NodeMeta and issubclass(obj, Node):
        debug("[encapsulate] obj is a subclass of Node")
        result = obj()
    elif callable(obj): 
        # 如果对象是可调用对象（例如函数或者实现 __call__ 方法的类），创建一个 Node 实例
        debug("[encapsulate] obj is a callable object")
        class SimpleNode(Node):
            node_id = NodeType.WRAPPED
            def operate(
                self, input: Any, config: Optional[RT] = None, *args, **kwargs
            ) -> NodeBus:
                return NodeBus(NodeType.WRAPPED, obj(input, config, *args, **kwargs))
        result = SimpleNode()
    elif isinstance(obj, Dict) or isinstance(obj, str): 
        # 如果对象是字典或字符串，创建一个Node实例
        debug("[encapsulate] obj is a dict or str")
        class SimpleNode(Node):
            node_id = NodeType.WRAPPED
            def operate(
                input: Optional[Any] = None, config: RT = None, *args, **kwargs
            ) -> NodeBus:
                return NodeBus(NodeType.WRAPPED, obj)
        result = SimpleNode()
    else:
        # 如果对象不是 Node 或可调用对象，直接返回一个简单的 Node 实例
        debug("[encapsulate] obj is a simple object")
        class SimpleNode(Node):
            node_id = NodeType.WRAPPED
            def operate(
                self, input: Any, config: Optional[RT] = None, *args, **kwargs
            ) -> NodeBus:
                if hasattr(obj, "operate"):  # 如果对象有 operate 方法，则调用该方法
                    return NodeBus(NodeType.WRAPPED, obj.operate(input, config, *args, **kwargs))
                else:
                    return NodeBus(NodeType.WRAPPED, obj)  #  否则返回对象本身
        result = SimpleNode()
    return result


if __name__ == "__main__":
    # python -m catena.catena_core.node.base
    pipe = Node >> Node