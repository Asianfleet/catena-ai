from __future__ import annotations
import asyncio
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional, 
    Union
)

from .meta import NodeMeta
from .completion import NodeBus, NodeCompletion
from ..alias.builtin import NodeType
from ..callback.node_callback import (
    NodeCallback, 
    NodeCallbackRegister as register,
    node_callback
)
from ...error.chainerr import *
from ...settings import (
    settings, 
    RTConfig as RT
)
from ...cli.tools import debug, warning

class NodeExtraMetaData(BaseModel):
    """ Node 的元数据类 """

    # 前置节点
    former_node: Optional[str] = None             
    # 后置节点
    latter_node: Optional[str] = None             
    # 节点位置
    position: Optional[str] = None                
    # 是否显示输入数据
    display_input: bool = True          
    # 节点索引
    Index: int = 0                      
    # 输入数据
    Input: Optional[NodeCompletion] = None                   
    # 输出数据
    Output: Optional[NodeCompletion] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Node(BaseModel):
    """
    链式调用的核心基类（抽象基类），是链中每个节点的基本构建块。任何类可通过继承该类并实现内部接口成为一个节点。
    该基类提供了调试可视化、回调注册和执行等功能：
     - 支持通过 visualize 装饰器对任务执行过程进行调试和实时可视化，允许开发者在任务执行时查看输入输出的变化。
     - 支持通过链式调用 (__or__, __rshift__ 等运算符) 结合 NodeChain 将多个任务组合成一个执行流，形成任务链。
     - 每个 Node 对象都可以执行具体的任务逻辑，通过实现 operate 方法将输入数据传递并返回处理结果。
     - 通过回调机制，用户可以在任务执行过程中进行动态的操作与数据交互。
    """
    
    # 节点ID
    node_id: Enum = Field(
        default=NodeType.UDFN, 
        init=False,
        description="节点ID"
    )
    # 节点样式
    style: Optional[Union[str, Enum]] = Field(
        default=None,
        init=False,
        metadata={"description": "节点样式"}
    )                                      
    # 节点是否可重复出现
    # 如果为True，则表示节点可以重复出现，否则不允许重复出现。
    repeatable: bool = Field(
        default=False,
        init=False,
        metadata={"description": "节点是否可重复出现"}
    )
    
    # 无需手动初始化
    # 节点元数据
    extra_metadata: NodeExtraMetaData = Field(
        default_factory=NodeExtraMetaData, 
        init=False,
        description="节点元数据"
    )
    # 关联的链的id
    related_chain_id_list: List[Enum] = Field(default_factory=list, init=False)         
    # 回调函数注册表
    callback_register: Dict[str, Callable] = Field(default_factory=dict, init=False)
  
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def nid(self):
        if self.node_id == NodeType.UDFN:
            self.node_id = self.signature
        return self.node_id
    
    @property
    def signature(self):
        return self.__class__.__name__
    
    @property
    def callbacks(self):
        return self.callback_register
    
    @property
    def related_chain_ids(self):
        return self.related_chain_id_list
    
    @related_chain_ids.setter
    def related_chain_ids(self, value: List[Enum]):
        self.related_chain_id_list = value
    
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
        return NodeChain(first=self, second=encapsulate(other))
       
    def __ror__(self, other: Any) -> NodeChain:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(first=encapsulate(other), second=self)
     
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
        return NodeChain(first=self, second=encapsulate(other))
       
    def __rrshift__(self, other: Any) -> NodeChain:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(first=encapsulate(other), second=self)
    
    def __call__(self, other):
        return self.__rshift__(other)

    @node_callback("default")
    def _callback(self, input: NodeCallback) -> Any:
        pass

    def _operate(self, input: Any, runtime: RT = None) -> Any:
        return NotImplementedError

    def operate(self, input: NodeCompletion) -> NodeCompletion:
        """ 提供给外部的接口 """
        if not isinstance(input, NodeCompletion):
            completion = NodeCompletion()
            completion.update(main_data=input)
        else:
            completion = input
        
        main_input = completion.main_data
        runtime = completion.extra_data
        
        output = self._operate(main_input, runtime)
        
        node_completion = NodeCompletion(
            type=self.node_id,
            main_data=output,
            extra_data=runtime
        )
        
        return node_completion

    def to_dict(self) -> Dict[str, Any]:
        """ 将 Node 对象转换为字典 """
        _dict = self.model_dump()
        
        # 处理枚举类型的序列化
        from enum import Enum
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict
        
class NodeChain(BaseModel):
    """
    是 Node 的具体实现，主要用于将多个 Node 任务串联起来，形成一个按顺序执行的任务链：
     - 每个任务的输出会自动作为下一个任务的输入，实现任务的顺序执行。
     - 提供了灵活的链式调用能力，可以将多个 Node 任务组合在一起执行，形成复杂的任务流水线。
     - 在执行过程中，NodeChain 会确保每个任务按顺序执行，并处理回调机制与配置。
     - 支持链式操作符，允许开发者方便地将多个任务组合在一起。
     - operate 方法会依次执行链中的所有任务，并最终返回最后一个任务的输出结果。
    """
    # 需要手动初始化
    # 形成任务链的第一个节点
    first: Node
    # 第二个节点
    second: Node
    
    # 下面为内部属性，不需要手动初始化
    # 任务链
    chain: List[Node] = Field(default_factory=list, init=False)
    # 是否已编译
    compiled: bool = Field(default=False, init=False)
    # 任务总线
    Bus: NodeBus = Field(default_factory=NodeBus, init=False)
    # 链中每个元素id的存储列表
    chain_id: List[Enum] = Field(default_factory=list, init=False)
    # 链中每个元素的类名列表
    chain_name: List[str] = Field(default_factory=list, init=False)
    # 链中每个元素的签名（signature）存储列表
    chain_sig: List[str] = Field(default_factory=list, init=False)
    # 支持自定义类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.chain = [self.first, self.second]  
        
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
    ) -> NodeChain:
        return self.__call__(other)

    def __rshift__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
            type
        ],
    ) -> NodeChain:
        return self.__call__(other)
    
    def __rrshift__(self, other: Any) -> NodeChain:
        self.chain = [encapsulate(other)] + self.chain
        return self

    # TODO: 添加节点之间的连接关系检查机制
    def compile(self):
        """ “编译”任务链，确保任务链的正确性 """
        
        if not hasattr(self, "compiled") or not self.compiled:
            try:
                # 确保链中至少包含两个节点
                if len(self.chain) < 2:
                    raise ChainCompileError("Chain must have at least two nodes")
                # 设置节点的位置
                self.chain[0].extra_metadata.position = "start"
                self.chain[-1].extra_metadata.position = "end"
                chain_middle = self.chain[1:-1]
                for node in chain_middle:
                    node.extra_metadata.position = "middle"
                
                # 确保节点的继承关系
                for node in self.chain:
                    if node.__class__.mro().index(Node) > 2:
                        raise ChainCompileError(
                            "Node classe allow at most second-level inheritance"
                        )
                self.chain_id: List = [node.nid for node in self.chain]
                self.chain_name: List = [node.signature for node in self.chain]
                self.chain_sig: List = [node.signature for node in self.chain]
                debug(self.chain_id)
                debug(self.chain_name)
                debug(self.chain_sig)
                # 确保节点类的唯一性
                if (
                    len(self.chain_id) != len(set(self.chain_id)) 
                    or 
                    len(self.chain_sig) != len(set(self.chain_sig))
                    or 
                    len(self.chain_name) != len(set(self.chain_name))
                ):
                    raise ChainCompileError("These properties of Node must be unique")
            except ChainCompileError as e:
                e.trace()
            
            # 设置节点的连接关系
            self.chain[0].extra_metadata.latter_node = self.chain[1].signature
            self.chain[-1].extra_metadata.former_node = self.chain[-2].signature
            if chain_middle:
                for idx in range(1, len(chain_middle) + 1):
                    self.chain[idx].extra_metadata.former_node = self.chain[idx - 1].signature
                    self.chain[idx].extra_metadata.latter_node = self.chain[idx + 1].signature
            for node in self.chain:
                node.related_chain_ids = self.chain_id
            
            self.compiled = True

    def handle_callback(self, node: Node):
        """ 在一个节点运行之后调用回调函数 """
        
        # 1、判断节点位置
        # 如果节点不是起始节点，就有可能调用上一个节点的回调函数，反之则无需处理
        if node in self.chain[1:]:
            debug("[handle_callback] node:", node.signature)
            node_idx = node.extra_metadata.Index
            # 2、判断是否为数据节点
            # 如果不是纯数据节点（见 encapsulate 函数），则需要处理回调函数
            if node.extra_metadata.Output.type != NodeType.WRAPPED: 
                # 3、获取回调函数的参数
                cb_args = node.extra_metadata.Output.callback.dict
                # 4、获取回调函数的目标节点
                # 如果有该参数，则获取目标节点的索引，执行对应的回调函数
                if cb_args["target"]:   
                    try:
                        index = self.chain_id.index(cb_args["target"])
                    except ValueError:
                        debug("[handle_callback] target node not found, use previous node")
                        index = node_idx - 1
                        cb_args["name"] = "default"
                # 否则，获取上一个节点的索引，执行上一个节点的回调函数    
                else:   
                    index = node_idx - 1
                # 5、获取回调函数所在的节点
                target_node: Node = self.chain[index]
                # 6、执行对应的回调函数
                debug("[handle_callback] cb_args:", cb_args)
                cb_result: Union[NodeCompletion, Any] = register.run_callback(
                    target_node, 
                    cb_args["name"], 
                    cb_args["main_input"], 
                    *cb_args["args"], 
                    **cb_args["kwargs"]
                )
                debug("[handle_callback] callback result:", cb_result)
                # 7、处理回调输出
                # 规定：如果回调函数返回结果，则使用回调函数的返回结果作为输出
                if cb_result:   
                    if isinstance(cb_result, NodeCompletion):
                        self.Bus.latest.update(cb_result)
                    else:
                        self.Bus.latest.main_data = cb_result
                    display = True  # 此时下一个节点的输入就不再是上一个节点的输出，因此需要显示出来
                else:               
                    # 否则使用原输出
                    display = False # 此时下一个节点的输入就是上一个节点的输出，故不需要显示出来
                if node != self.chain[-1]:  # 只有在不是最后一个节点时才需要设置显示输入
                    debug("[handle_callback] display:", display)
                    self.chain[node_idx + 1].extra_metadata.display_input = display
            # 如果是数据节点，则直接使用原输出，不存在回调函数的情况
            else:
                pass
        # 如果是起始节点，则不需要处理回调函数
        else:
            pass
            
    def operate(self, input: Any = None, *args, **kwargs) -> NodeCompletion:
        
        # 编译整个链路，确保链路可执行
        self.compile()  
        debug("[NodeChain.operate] chain:\n", "\n".join([str(node) for node in self.chain]))
        # 将输出作为第一个 
        self.Bus.add(main_data=input, extra_data = RT())

        # 遍历链路中的节点
        for index, node in enumerate(self.chain):
            # 1、设置节点的索引
            node.extra_metadata.Index = index   
            # 2、设置节点的输入
            node.extra_metadata.Input = self.Bus.latest 
            # 3、执行节点，并把节点输出在 NodeBus 中更新
            node_completion = node.operate(self.Bus.latest)  
            self.Bus.add(node_completion)
            # 4、设置节点的输出
            node.extra_metadata.Output = node_completion
            debug(
                "[NodeChain.operate] node:", node.signature, 
                "output:", node.extra_metadata.Output
            )
            # 5、处理回调函数，更新 NodeBus
            self.handle_callback(node)
            
        return self.Bus.latest
    
    def print_operate(self, input: Any = None, *args, **kwargs) -> NodeCompletion:
        pass

    def to_dict(self) -> Dict[str, Any]:
        """ 将 NodeChain 对象转换为字典 """
        _dict = self.model_dump()
        
        # 处理枚举类型的序列化
        from enum import Enum
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict
    
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
        self, inputs: List[Any], config: Optional[RT] = None, *args, **kwargs
    ) -> List[NodeBus]:
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
    elif callable(obj): 
        # 如果对象是可调用对象（例如函数或者实现 __call__ 方法的类），创建一个 Node 实例
        debug("[encapsulate] obj is a callable object")
        class SimpleNode(Node):
            node_id: NodeType = NodeType.WRAPPED
            def operate(self, input: NodeCompletion) -> NodeCompletion:
                obj_input = input.main_data
                output = NodeCompletion(main_data=obj(obj_input), extra_data=RT())
                return output
        result = SimpleNode()
    elif isinstance(obj, (dict, str)): 
        # 如果对象是字典或字符串，创建一个Node实例
        debug("[encapsulate] obj is a dict or str")
        class SimpleNode(Node):
            node_id: NodeType = NodeType.WRAPPED
            def operate(self, input: Any) -> NodeCompletion:
                output = NodeCompletion(main_data=obj, extra_data=RT())
                return output
                
        result = SimpleNode()
    else:
        # 如果对象不是 Node 或可调用对象，直接返回一个简单的 Node 实例
        debug("[encapsulate] obj is a simple object")
        class SimpleNode(Node):
            node_id: NodeType = NodeType.WRAPPED
            def operate(self, input: NodeCompletion) -> NodeCompletion:
                if hasattr(obj, "operate"):  # 如果对象有 operate 方法，则调用该方法
                    result = obj.operate(input)
                    output = NodeCompletion(main_data=result, extra_data=RT())
                else:
                    output = NodeCompletion(main_data=obj, extra_data=RT())
                return output
        result = SimpleNode()
    return result


if __name__ == "__main__":
    # python -m catena.catena_core.node.base
    #pipe = Node() >> Node()
    pass