"""``langchain-core`` defines the base abstractions for the LangChain ecosystem.

The interfaces for core components like chat models, LLMs, vector stores, retrievers,
and more are defined here. The universal invocation protocol (Runnables) along with
a syntax for combining components (LangChain Expression Language) are also defined here.

No third-party integrations are defined here. The dependencies are kept purposefully
very lightweight.
"""

from __future__ import annotations
import copy
import asyncio
from enum import Enum
from rich.tree import Tree
from rich.live import Live
from functools import wraps
from rich.console import Console
from dataclasses import dataclass, field
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any, Dict, List,
    Optional, Union, Callable, Literal
)

from .catena_core.settings import settings, RTConfig, debug
from .catenasmith.cli_tools import (
    Style as sty,
    Formatter as fmt
)


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

@dataclass
class NodeCompletion:
    """ 存储每个节点的 operate 函数输出 """
    
    type: str
    main: Any = "None"
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    config: RTConfig = RTConfig()
    callback: Callback = Callback()
    
    @property
    def dict(self):
        """ 将 NodeCompletion 对象转换为字典 """
        return self.__dict__
        
    @property
    def list(self):
        """ 将 NodeCompletion 对象转换为列表 """
        prop_list = []
        for attr, value in self.__dict__.items():
            if value:
                if hasattr(value, "data"):
                    prop_list.append(attr + ": " + str(value.data))
                else:
                    prop_list.append(attr + ": " + str(value))
        return prop_list
    
class NodeCompletionType(Enum):
    """ 内置节点类型 """
    # 常量定义
    LLMIP = "LLM-input"
    LLMOP = "LLM-output"
    PMOP = "prompt-output"
    PAOP = "parsed-output"
    WRAPPED = "wrapped-output"   
    
class NodeType(Enum):
    """ 内置节点 ID  """
    LM = "llm"
    PM = "prompt"
    PS = "parser"
    MO = "memory"
    WP = "wrapper"
    
class NodeMeta(ABCMeta):
    """ Node 的元类，用于实现类本身之间的运算符重载和链式调用 """
    def __rshift__(cls, other):
        debug("[NodeMeta.__rshift__] type of cls:", type(cls))
        debug("[NodeMeta.__rshift__] type of other:", type(other))
        return cls() >> other
    
    def __rrshift__(cls, other):
        debug("[NodeMeta.__rrshift__] type of cls:", type(cls))
        debug("[NodeMeta.__rrshift__] type of other:", type(other))
        return other >> cls()
    
# TODO: 节点改进
# 1、完善并优化回调函数逻辑以及可视化输出（inspector）逻辑
# 2、增加流式输出处理机制
# 3、上下文缓存（Context Cache）？
class Node(ABC, metaclass=NodeMeta):
    """
    链式调用的核心基类（抽象基类），是链中每个节点的基本构建块。任何类可通过继承该类并实现内部接口成为一个节点。
    该基类提供了调试可视化、回调注册和执行等功能：
     - 支持通过 visualize 装饰器对任务执行过程进行调试和实时可视化，允许开发者在任务执行时查看输入输出的变化。
     - 支持通过链式调用 (__or__, __rshift__ 等运算符) 结合 NodeChain 将多个任务组合成一个执行流，形成任务链。
     - 每个 Node 对象都可以执行具体的任务逻辑，通过实现 operate 方法将输入数据传递并返回处理结果。
     - 通过回调机制，用户可以在任务执行过程中进行动态的操作与数据交互。
    """

    def __init__(self, style: str=None, signature: str=None):
        self._style = style or sty.BW   # 设置节点的样式
        self.callback_register = {} # 回调函数注册表
        self.display_input = True  # 是否显示输入数据
        self._signature = signature or self.type  # 设置节点的签名
        self._state = Tree(f"[{sty.fc(self._signature, self._style)}]") # 创建一个 Tree 对象
        self._inspector = Live(self._state, console=Console())    # 创建一个 Live 对象
        
    @property
    def type(self):
        return self.__class__.__name__
    
    @property
    def signature(self):
        return self._signature

    def update_state(self, *states):
        statespm = list(copy.deepcopy(states))
        debug("[update_state] statespm:", statespm)
        
        if statespm[0] == "Input:":
            if isinstance(statespm[1], list):
                for s in statespm[1]:
                    if isinstance(s, dict):
                        if "image" in s:
                            s["image"] = "base64_placeholder"
                            break        
                statespm = [s for s in statespm[1]]
            else:   # TODO:处理其他类型的情况
                if isinstance(statespm[1], dict):
                    if "image" in statespm[1]:
                        statespm[1]["image"] = "base64_placeholder"
                statespm = [str(statespm[1])]
            state_type = fmt.fc("Input:", sty.B)
            debug("[update_state] statespm[1:]:", statespm[1:])
            # 添加输入数据到状态树
            stype = self._state.add(state_type)
            for s in statespm:
                stype.add(s)
        elif statespm[0] == "Output:":  # 此时 statepm 只有两个元素
            assert isinstance(statespm[1], NodeCompletion), "Output must be a NodeCompletion object"
            completion:NodeCompletion = statespm[1] # 获取 NodeCompletion 对象
            if completion.type == NodeCompletionType.LLMIP: # 输出类型是大模型输入消息的列表
                for msg in completion.main: # 此时 main 是一个列表     
                    if msg["role"] == "user" and isinstance(msg["content"], list):  # 包含图像信息
                        for content in msg["content"]:
                            if "image_url" in content:
                                url = content["image_url"]["url"]
                                truncated_url = url[:30] + "..." + url[-10:]    # 截断图像base64编码
                                content["image_url"]["url"] = truncated_url
            elif completion.type == NodeCompletionType.LLMOP:  # 输出类型是大模型输出消息的字典
                completion.main = completion.main["content"]  # 只保留内容部分
            else:
                pass #TODO:更多自定义逻辑
            # OUTPUT 树节点
            state_type = fmt.fc("Output:", sty.B)
            # NodeCompletion 对象转为列表，为树的分支
            statespm = [s for s in completion.list]
            # 添加输出树节点到状态树
            stype = self._state.add(state_type)
            for s in statespm:
                stype.add(s)    # 添加分支
            # CallBack 树节点
            if completion.callback:
                # 添加回调函数节点到状态树
                callback_state = fmt.fc("Callback:", sty.B)
                statescb = [s for s in completion.callback.list]
                stype = self._state.add(callback_state)
                for s in statescb:
                    stype.add(s)
    
        self._inspector.update(self._state)

    def visualize(func):
        """ 装饰器函数，用于增强运行时的调试和可视化功能。 """
        @wraps(func)
        def wrapper(self, input: Union[str, dict], config: RTConfig = None, *args, **kwargs):
            """
            装饰器的包装函数，负责实际的调试和可视化逻辑。
            
            参数:
            - self: Node类的实例。
            - input: 运行时输入，可以是字符串或字典。
            - config: 运行时配置，用于控制行为。
            - *args, **kwargs: 其他传递给被装饰函数的参数。
            
            返回:
            - result: 被装饰函数的执行结果。
            """
            # 检查是否禁用了可视化调试
            config = config or RTConfig()
            input = input or self.input
            if (config().get("enable_chain_visualize", None) or 
                not settings.debug.enable_chain_visualize):
                print("[Node.visualize] 调试关闭")
                result = func(self, input, config, *args, **kwargs)
            else:
                try:
                    debug("[Node.visualize] node:", self.type)
                    # 创建一个 Tree 以及对象，用于显示状态信息
                    if self._latter_node:
                        thisnode = fmt.fc(self._signature, self._style)
                        nextnode = fmt.fc(self._latter_node, sty.GR_L)
                        root = f"[{thisnode} -> {nextnode}]"
                    else:
                        root = f"[{fmt.fc(self._signature, self._style)}]"
                    self._state = Tree(root)
                    # 启动Live调试工具
                    self._inspector.start()  # 开始 Live
                    # 更新状态显示输入数据
                    debug("[Node.visualize] display_input:", self.display_input)
                    if self.display_input:
                        self.update_state("Input:", input)
                    else:
                        self.update_state("Input:", self._former_node + ".Output.main") 
                    # 执行被装饰的函数
                    result:NodeCompletion = func(self, input, config, *args, **kwargs)
                    # 更新状态显示输出结果
                    self.update_state("Output:", result)
                finally:
                    # 无论成功与否，确保停止Live调试工具
                    self._inspector.stop()  # 停止 Live
            return result
        return wrapper

    def callback(register_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, main_input, *args, **kwargs):
                if register_name in self.callback_register:
                    raise RuntimeError("Callback name already exists")
                else:
                    self.callback_register[register_name] = func
                    return func(main_input, *args, **kwargs)    # 调用被装饰的函数
            return wrapper
        return decorator

    def run_callback(
        self, register_name: str = None, main_input: Any=None, *args, **kwargs
    ):
        """ 运行指定的回调函数 """
        if register_name:
            if register_name in self.callback_register:
                return self.callback_register[register_name](main_input, *args, **kwargs)
            else:
                raise KeyError("Callback name does not exist")
        else:
            if "default_callback" in self.callback_register:
                return self.callback_register["default_callback"](main_input, *args, **kwargs)
    
    @abstractmethod
    def operate( # TODO: 修改配置传入逻辑，常用的核心设置写成一个专门的设置类
        self, input: Any, config: Optional[RTConfig] = None, *args, **kwargs
    ) -> NodeCompletion:
        pass

    # TODO: 将__or__和__ror__的重载功能进行替换 
    def __or__(
        self,
        other: Union[
            Node,
            Callable[[Any], Any],
            Dict[str, Any],
        ],
    ) -> Node:
     
        return NodeChain(self, other)
       
    def __ror__(self, other: Any) -> NodeChain:
        # 将非Node对象转换为Node对象，并创建一个NodeChain
        if isinstance(other, NodeChain):
            return other(self)
        else:
            return NodeChain(encapsulate(other), self)
     
    """ 
    ############################# 链式调用的核心原理：运算符重载 #################################
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
        ],
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

class NodeChain(Node):
    """
    是 Node 的具体实现，主要用于将多个 Node 任务串联起来，形成一个按顺序执行的任务链：
     - 每个任务的输出会自动作为下一个任务的输入，实现任务的顺序执行。
     - 提供了灵活的链式调用能力，可以将多个 Node 任务组合在一起执行，形成复杂的任务流水线。
     - 在执行过程中，NodeChain 会确保每个任务按顺序执行，并处理回调机制与配置。
     - 支持链式操作符，允许开发者方便地将多个任务组合在一起。
     - operate 方法会依次执行链中的所有任务，并最终返回最后一个任务的输出结果。
    """
    
    def __init__(
        self, former: Node, latter: Node, *args, **kwargs
    ) -> None:
        self.chain = [{"node": former}, {"node": latter}]
        self.console = Console()

    def __call__(self, node):
        self.chain.append({
            "node": encapsulate(node)
        })
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
            assert len(self.chain) > 1, "Chain must have at least 2 elements"
            self.name_list = [type(node["node"]) for node in self.chain]
            self.id_list = [node["node"].id for node in self.chain]
            debug("[compile] ", self.name_list)
            assert len(self.name_list) == len(set(self.name_list)), "Chain must have unique node names"
            assert len(self.id_list) == len(set(self.id_list)), "Chain must have unique node ids"
            
            for idx, node in enumerate(self.chain):
                if idx != 0:
                    node["node"].__setattr__("_former_node", self.chain[idx - 1]["node"].type)  
                else:
                    self.chain[0]["node"].__setattr__("_pos", "start")      
                    node["node"].__setattr__("_former_node", None)              
                if idx != len(self.chain) - 1:
                    node["node"].__setattr__("_latter_node", self.chain[idx + 1]["node"].type)
                    if idx != 0:
                        node["node"].__setattr__("_pos", "middle")
                else:
                    self.chain[-1]["node"].__setattr__("_pos", "end")
                    node["node"].__setattr__("_latter_node", None)
                
                debug("[compile] _pos:", node["node"]._pos) 
                debug("[compile] _former_node:", node["node"]._former_node)
                debug("[compile] _latter_node:", node["node"]._latter_node)
            
            self.compiled = True

    def handle_callback(self, node: dict):
        """ 调用回调函数 """
        node_output: NodeCompletion = node["Output"]
        debug("[handle_callback] node:", node["node"].type)
        if node_output.type != NodeCompletionType.WRAPPED: # 若不是封装节点（见 encapsulate 函数），则需要处理回调函数
            # 获取回调函数的参数
            cb_args = node_output.callback.data
            # 获取回调函数的目标节点
            if cb_args["target"]:
                index = self.id_list.index(cb_args["target"])
            else:
                index = node["Index"] - 1
            target_node = self.chain[index]["node"]
            cb_result = target_node.run_callback(
                cb_args["name"], cb_args["main_input"], *cb_args["args"], **cb_args["kwargs"]
            )
            debug("[handle_callback] callback result:", cb_result)
            if cb_result:   # 规定：如果回调函数返回结果，则使用回调函数的返回结果作为输出
                output = cb_result.main if isinstance(cb_result, NodeCompletion) else cb_result
                display = True  # 此时下一个节点的输入就不再是上一个节点的输出，因此需要显示出来
            else:           # 否则使用原输出
                output = node_output.main
                display = False # 此时下一个节点的输入就是上一个节点的输出，故不需要显示出来
            debug("[handle_callback] _pos:", node["node"]._pos)
            if node["node"]._pos != "end":  # 只有在不是最后一个节点时才需要设置显示输入
                debug("[handle_callback] display:", display)
                self.chain[node["Index"] + 1]["node"].__setattr__("display_input", display)
        else:
            output = node_output.main   # 若为封装节点，则直接使用原输出
        return output

    def operate(
        self, input: Optional[Any] = None, config: RTConfig = None, *args, **kwargs
    ) -> NodeCompletion:
        
        """ def console_dec(stat: Literal["Activated", "Stopped"]):
            if self.config().get("enable_chain_visualize", None) or settings.debug.enable_chain_visualize:
                if stat == "Activated":
                    color = settings.style.chain_start
                else:
                    color = settings.style.chain_end
                    
                spacer = settings.style.chain_spacer
                rep = settings.style.spacer_repeat
                self.console.print(fmt.fc(f"{spacer*rep} Chain {stat} {spacer*rep}", color)) """
            
        self.compile()  # 编译任务链
        self.config = config or RTConfig()
        self.data = [input]
        # 输出管道开始信息
        #console_dec("Activated")
        for index, node in enumerate(self.chain):
            output: NodeCompletion = node["node"].operate(self.data[-1], self.config, *args, **kwargs)
            node["Index"] = index
            node["Input"] = self.data[-1]
            node["Output"] = output
            self.data.append(self.handle_callback(node))
            self.config = output.config
        # 显示管道结束信息
        #console_dec("Stopped")
        
        return self.data[-1]
    
    def operate(
        self, input: Optional[Any] = None, config: RTConfig = None, *args, **kwargs
    ) -> NodeCompletion:
        
        """ def console_dec(stat: Literal["Activated", "Stopped"]):
            if self.config().get("enable_chain_visualize", None) or settings.debug.enable_chain_visualize:
                if stat == "Activated":
                    color = settings.style.chain_start
                else:
                    color = settings.style.chain_end
                    
                spacer = settings.style.chain_spacer
                rep = settings.style.spacer_repeat
                self.console.print(fmt.fc(f"{spacer*rep} Chain {stat} {spacer*rep}", color)) """
            
        self.compile()  # 编译任务链
        self.config = config or RTConfig()
        self.data = [input]
        # 输出管道开始信息
        #console_dec("Activated")
        for index, node in enumerate(self.chain):
            output: NodeCompletion = node["node"].operate(self.data[-1], self.config, *args, **kwargs)
            node["Index"] = index
            node["Input"] = self.data[-1]
            node["Output"] = output
            self.data.append(self.handle_callback(node))
            self.config = output.config
        # 显示管道结束信息
        #console_dec("Stopped")
        
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
    
    def operate(self, input, config: Optional[RTConfig] = None, *args, **kwargs):
        if self.chain:
            return self.chain.operate(input, config, *args, **kwargs)
        else:
            raise ValueError("No chain has been set")  # 确保链已被设置
    
    async def ainvoke(
        self, inputs: list[Any], config: Optional[RTConfig] = None, *args, **kwargs
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
                self, input: Any, config: Optional[RTConfig] = None, *args, **kwargs
            ) -> NodeCompletion:
                return NodeCompletion(NodeCompletionType.WRAPPED, obj(input, config, *args, **kwargs))
        result = SimpleNode()
    elif isinstance(obj, Dict) or isinstance(obj, str): 
        # 如果对象是字典或字符串，创建一个Node实例
        debug("[encapsulate] obj is a dict or str")
        class SimpleNode(Node):
            id = NodeType.WP
            def operate(
                input: Optional[Any] = None, config: RTConfig = None, *args, **kwargs
            ) -> NodeCompletion:
                return NodeCompletion(NodeCompletionType.WRAPPED, obj)
        result = SimpleNode()
    else:
        # 如果对象不是 Node 或可调用对象，直接返回一个简单的 Node 实例
        debug("[encapsulate] obj is a simple object")
        class SimpleNode(Node):
            id = NodeType.WP
            def operate(
                self, input: Any, config: Optional[RTConfig] = None, *args, **kwargs
            ) -> NodeCompletion:
                if hasattr(obj, "operate"):  # 如果对象有 operate 方法，则调用该方法
                    return NodeCompletion(NodeCompletionType.WRAPPED, obj.operate(input, config, *args, **kwargs))
                else:
                    return NodeCompletion(NodeCompletionType.WRAPPED, obj)  #  否则返回对象本身
        result = SimpleNode()
    return result
    
if __name__ == "__main__":
    # python -m src.modules.agent.core
   
    class LLMBaseMemory(Node):

        @abstractmethod
        def add(self, message):
            pass

        @abstractmethod    
        def pop(self):
            pass
        
        @abstractmethod
        def clear(self):
            pass

        @abstractmethod
        def run_callback(self):
            print("llm memory callback")
            pass

        def _or(self, other: Node[Any, Any]) -> Node[Any, Any]:
            print("llm memory or")
            if isinstance(other, LLMBaseMemory):
                return NodeChain(self, other)
            #else:
            #    return NodeChainWithBackforeward(self, other)
        
    class LLMChatMemory(LLMBaseMemory):
        """
        This class is used to store the chat history of the agent.
        """

        def __init__(self):
            self.messages = []

        def add(self, message):
            self.messages.append(message)

        def pop(self):
            self.messages = self.messages[:-2]
        
        def clear(self):
            self.messages = []

        def operate(self, input):
            print("chat memory operate")
            self.add(input)
            print("messages:", self.messages)
            return self.messages

        def run_callback(self, message):
            print("chat memory callback")
            self.add(message)

    class pm(Node):

        def __init__(self):
            pass

        def operate(self, input: str) -> str:
            print("pm operate")
            return "prompt: " + input

    class ChatMistralAI(Node):
        def __init__(self):
            pass

        def operate(self, input: List) -> str:
            # 模拟API调用返回结果
            print("chat mistral ai operate")
            
            return input.append("reply: I'm fine, thank you")
    
    class StrOutputParser:
        def parse(self, input: str):
            return input + "_parsed"

    class LLMOutputParser():

        def __init__(self, parser):
            self.parser = parser
            self.node = None

        def operate(self, input: str):
            print("LLMOutputParser operate")
            return self.parser.parse(input)
        
    """ print("llm memory test")
    llmparser = LLMOutputParser(StrOutputParser())
    pipe = pm() | LLMChatMemory() | ChatMistralAI() | llmparser

    print("result:", pipe.operate("hello")) """  
    if Callback():
        print("None")