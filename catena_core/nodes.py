from __future__ import annotations
import asyncio
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any, Dict, List, 
    Optional, Union, Callable
)

from .settings import (
    settings, 
    RTConfig as RT,
    debug
)
from .error.chainerr import *
from .alias.builtin import NodeCompletionType, NodeType
from .callback.node_callbacks import (
    Callback, 
    NodeCallbackRegister as Register
)


class NodeMeta(ABCMeta):
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
class NodeCompletion:
    """ 存储每个节点的 operate 函数输出 """
    
    type: Enum   # 节点输出类型
    main: Any = "None"  # 主输出
    args: tuple = ()    # 位置参数
    kwargs: dict = field(default_factory=dict)  # 关键字参数
    config: RT = RT()   # 运行时配置
    callback: Callback = Callback() # 回调函数配置
    
    @property
    def dict(self):
        """ 将 NodeCompletion 对象转换为字典 """
        return self.__dict__
        
    @property
    def list(self):
        """ 将 NodeCompletion 对象的非空属性转换为列表 """
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

class Node(ABC, metaclass=NodeMeta):
    """
    链式调用的核心基类（抽象基类），是链中每个节点的基本构建块。任何类可通过继承该类并实现内部接口成为一个节点。
    该基类提供了调试可视化、回调注册和执行等功能：
     - 支持通过 visualize 装饰器对任务执行过程进行调试和实时可视化，允许开发者在任务执行时查看输入输出的变化。
     - 支持通过链式调用 (__or__, __rshift__ 等运算符) 结合 NodeChain 将多个任务组合成一个执行流，形成任务链。
     - 每个 Node 对象都可以执行具体的任务逻辑，通过实现 operate 方法将输入数据传递并返回处理结果。
     - 通过回调机制，用户可以在任务执行过程中进行动态的操作与数据交互。
    """

    # 不能在外部更改的属性
    _id: str = NodeType.UD                          # 节点ID
    _style: Union[str, Enum] = None                 # 节点样式
    _input_type: type = Any                         # 接受的输入类型
    _related_chain_ids: List[Enum] = []              # 关联的链的id
    _callback_register: Dict[str, callable] = {}    # 回调函数注册表
       
    # 可在外部更改的属性
    former_node: str = None                     # 前置节点
    latter_node: str = None                     # 后置节点
    position: str = None                        # 节点位置
    display_input: bool = True                  # 是否显示输入数据
    Index: int = 0                              # 节点索引
    Input: Any = None                           # 输入数据
    Output: Optional[NodeCompletion] = None     # 输出数据
    
    def __init__(
        self, style: Union[str, Enum] = None, signature: str = None
    ):
        self._style = style or self._style
        self._signature = signature
    
    @property
    def id(self):
        return self._id
    
    @property
    def input_type(self):
        return self._input_type
    
    @property
    def type(self):
        return self.__class__.__name__
    
    @property
    def style(self):
        if isinstance(self._style, Enum):
            return self._style.value
        else:
            return self._style
    
    @property
    def signature(self):
        if self._signature is None:
            self._signature = self.type
        return self._signature
    
    @property
    def callback_register(self):
        return self._callback_register
    
    @property
    def related_chain_ids(self):
        return self._related_chain_ids
    
    @related_chain_ids.setter
    def related_chain_ids(self, chain_ids: List[Enum]):
        self._related_chain_ids = chain_ids
      
    # TODO: 将__or__和__ror__的重载功能进行替换 
    def __or__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
            type
        ]
    ) -> Any:
        return NodeChain(self, encapsulate(other))
       
    def __ror__(self, other: Any) -> Any:
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
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
            type
        ]
    ) -> Node:
     
        return NodeChain(self, encapsulate(other))
       
    def __rrshift__(self, other: Any) -> Node:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(encapsulate(other), self)
    
    def __call__(self, other):
        return self.__rshift__(other)

    @Register.node_callback("default")
    def _callback(self, main_input: Any, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def operate( 
        self, input: Any, config: Optional[RT] = None, *args, **kwargs
    ) -> NodeCompletion:
        """  """
        pass

class NodeChain(Node):
    """
    是 Node 的具体实现，主要用于将多个 Node 任务串联起来，形成一个按顺序执行的任务链：
     - 每个任务的输出会自动作为下一个任务的输入，实现任务的顺序执行。
     - 提供了灵活的链式调用能力，可以将多个 Node 任务组合在一起执行，形成复杂的任务流水线。
     - 在执行过程中，NodeChain 会确保每个任务按顺序执行，并处理回调机制与配置。
     - 支持链式操作符，允许开发者方便地将多个任务组合在一起。
     - operate 方法会依次执行链中的所有任务，并最终返回最后一个任务的输出结果。
    """
    
    def __init__(self, first: Node, second: Node):
        self._first = first
        self._second = second
        self.chain = [self._first, self._second]  
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
    ) -> Node:
        return self.__call__(other)

    def __rshift__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
            type
        ],
    ) -> Node:
        return self.__call__(other)

    # TODO: 添加节点之间的连接关系检查机制
    def compile(self):
        """ “编译”任务链，确保任务链的正确性 """
        
        if not hasattr(self, "compiled") or not self.compiled:
            try:
                if len(self.chain) < 2:
                    raise ChainCompileError("Chain must have at least two nodes")
                
                self.chain[0].position = "start"
                self.chain[-1].position = "end"
                chain_middle = self.chain[1:-1]
                for node in chain_middle:
                    node.position = "middle"
                self.chain_mro = [node.__class__.mro() for node in self.chain]
                self.chain_id = [node.id for node in self.chain]
                debug("[NodeChain.compile] ", self.chain_mro)
        
                if len(self.chain_id) != len(set(self.chain_id)):
                    raise ChainCompileError("Chain must have unique node ids")
            except ChainCompileError as e:
                e.trace()
            
            self.chain[0].latter_node = self.chain[1].signature
            self.chain[-1].former_node = self.chain[-2].signature
            if chain_middle:
                for idx in range(1, len(chain_middle) + 1):
                    self.chain[idx].former_node = self.chain[idx - 1].signature
                    self.chain[idx].latter_node = self.chain[idx + 1].signature
            for node in self.chain:
                node.related_chain_ids = self.chain_id
            
            self.compiled = True

    def handle_callback(self, node: Node):
        """ 在一个节点运行之后调用回调函数 """
        
        # 如果不是第一个节点，则有可能调用上一个节点的回调函数
        if node in self.chain[1:]:
            debug("[handle_callback] node:", node.type)
            # 若不是封装节点（见 encapsulate 函数），则需要处理回调函数
            if node.Output.type != NodeCompletionType.WRAPPED: 
                # 获取回调函数的参数
                cb_args = node.Output.callback.data
                # 获取回调函数的目标节点
                if cb_args["target"]:   # 如果有该参数，则获取目标节点的索引，执行对应的回调函数
                    index = self.chain_id.index(cb_args["target"])
                else:   # 否则，获取上一个节点的索引，执行上一个节点的回调函数
                    index = node.Index - 1
                # 获取回调函数所在的节点
                target_node: Node = self.chain[index]
                # 执行对应的回调函数
                debug("[handle_callback] cb_args:", cb_args)
                cb_result = Register.run_callback(
                    target_node, 
                    cb_args["name"], 
                    cb_args["main_input"], 
                    *cb_args["args"], 
                    **cb_args["kwargs"]
                )
                debug("[handle_callback] callback result:", cb_result)
                # 规定：如果回调函数返回结果，则使用回调函数的返回结果作为输出
                if cb_result:   
                    output = cb_result.main if isinstance(cb_result, NodeCompletion) else cb_result
                    display = True  # 此时下一个节点的输入就不再是上一个节点的输出，因此需要显示出来
                else:               # 否则使用原输出
                    output = node.Output.main
                    display = False # 此时下一个节点的输入就是上一个节点的输出，故不需要显示出来
                if node != self.chain[-1]:  # 只有在不是最后一个节点时才需要设置显示输入
                    debug("[handle_callback] display:", display)
                    self.chain[node.Index + 1].display_input = display
            else:
                output = node.Output.main   # 若为封装节点，则直接使用原输出，不存在回调函数的情况
            return output
        return node.Output.main

    def operate(
        self, input: Optional[Any] = None, config: RT = None, *args, **kwargs
    ) -> NodeCompletion:
    
        self.compile()  # 编译任务链
        self.config = config or RT()
        self.data = [input]
    
        for index, node in enumerate(self.chain):
            output: NodeCompletion = node.operate(self.data[-1], self.config, *args, **kwargs)
            node.Index = index
            node.Input = self.data[-1]
            node.Output = output
            debug("[NodeChain.operate] node:", node.type, "output:", output)
            self.data.append(self.handle_callback(node))
            self.config = output.config

        return self.data[-1]
    
#TODO: 完整实现异步节点逻辑
class NodeParallelNode(Node):
    """
    是 Node 的一个具体实现，能支持并行执行多个任务，并且可以与其他 Node 任务组成任务链：
    - 通过 operate 和 ainvoke 方法提供同步与异步执行模式。但主要用于在并行任务执行中处理输入数据并返回多个结果。
    - 任务并行化：该类的关键特性是并行执行任务。在异步模式下，ainvoke 方法会接收多个输入，使用 asyncio 和 ThreadPoolExecutor 来控制并发执行任务，
      并通过信号量 (asyncio.Semaphore) 控制并发数，从而提高效率。
    - 链式调用支持：与其他 Node 类一样，NodeParallelNode 也支持链式调用，通过 __rshift__ 和 __rrshift__ 方法将多个任务连接成一个执行流。
      每个任务的输出可以作为下一个任务的输入，形成一个动态任务链。
    - 灵活的输入与输出：该类支持不同的输入类型，既可以单个输入，也可以是多个输入，并支持通过并行方式处理多个输入数据。
      异步方法 ainvoke 返回多个结果，适合高并发任务场景。
    - 与其他 Node 类协同工作：NodeParallelNode 旨在与其他 Node 类（例如 NodeChain）一起工作，可以作为一个并行节点与顺序执行的任务链
      协同运作，从而满足复杂的任务流水线需求。
    """
    
    #TODO: 还未完成测试
    ####################################### 使用示例 ############################################
    # chain = [input1, input2, ..., inputN] >> NodeParallelNode()
    # chain.ainvoke()
    #
    # chain = input >> a(同步) >> b(同步，但输出列表) >> NodeParallelNode() >> c(同步，接受列表)
    # chain.ainvoke()
    #############################################################################################
    
    def __init__(self):
        self.chain = []
    
    def __rshift__(self, other):
        if not self.chain:
            self.chain = other # 确保第一个元素是 Node 对象
        else:
            self.chain = self.chain(other) # 说明 self.chain 是一个 NodeChain 对象
        return self
    
    def __rrshift__(self, other): # 应对 a >> NodeParallelNode， 但 a 不是 Node 对象
        self.input = other
        return self
    
    def operate(self, input, config: Optional[RT] = None, *args, **kwargs):
        if self.chain:
            return self.chain.operate(input, config, *args, **kwargs)
        else:
            raise ValueError("No chain has been set")  # 确保链已被设置
    
    async def ainvoke(
        self, inputs: list[Any], config: Optional[RT] = None, *args, **kwargs
    ) -> list[NodeCompletion]:
        """ 并行执行多个输入 """
        func = self.operate
        semaphore = asyncio.Semaphore(kwargs.get("concurrency", settings.base.concurrecy_limit))
        
        # 创建一个线程池，用于运行同步函数
        loop = asyncio.get_event_loop()
        tp_executor = ThreadPoolExecutor()

        async def inv_task(func, input, config, *args, **kwargs):
            async with semaphore:
                # 使用 run_in_executor 在线程池中运行同步函数
                return await loop.run_in_executor(tp_executor, func, input, config, *args, **kwargs)
            
        # 为每个输入创建一个任务
        invoke_tasks = [inv_task(func, input, config, *args, **kwargs) for input in inputs]

        # 并发运行任务并等待所有结果
        results = await asyncio.gather(*invoke_tasks)
        #logger.info("并发运行结束")

        # 关闭线程池
        tp_executor.shutdown(wait=True)
        
        return results

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
            id = NodeType.WP
            def operate(
                self, input: Any, config: Optional[RT] = None, *args, **kwargs
            ) -> NodeCompletion:
                return NodeCompletion(NodeCompletionType.WRAPPED, obj(input, config, *args, **kwargs))
        result = SimpleNode()
    elif isinstance(obj, Dict) or isinstance(obj, str): 
        # 如果对象是字典或字符串，创建一个Node实例
        debug("[encapsulate] obj is a dict or str")
        class SimpleNode(Node):
            id = NodeType.WP
            def operate(
                input: Optional[Any] = None, config: RT = None, *args, **kwargs
            ) -> NodeCompletion:
                return NodeCompletion(NodeCompletionType.WRAPPED, obj)
        result = SimpleNode()
    else:
        # 如果对象不是 Node 或可调用对象，直接返回一个简单的 Node 实例
        debug("[encapsulate] obj is a simple object")
        class SimpleNode(Node):
            id = NodeType.WP
            def operate(
                self, input: Any, config: Optional[RT] = None, *args, **kwargs
            ) -> NodeCompletion:
                if hasattr(obj, "operate"):  # 如果对象有 operate 方法，则调用该方法
                    return NodeCompletion(NodeCompletionType.WRAPPED, obj.operate(input, config, *args, **kwargs))
                else:
                    return NodeCompletion(NodeCompletionType.WRAPPED, obj)  #  否则返回对象本身
        result = SimpleNode()
    return result