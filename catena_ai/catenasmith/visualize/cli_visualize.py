import copy
from functools import wraps
from rich.tree import Tree
from rich.live import Live
from rich.status import Status
from rich.console import Console, Group
from abc import ABC, abstractmethod
from typing import (
    Any, 
    Callable, 
    List, 
    Optional, 
    Union
)

from ...cli.tools import (
    debug,
    info,
    Style as sty,
    Formatter as fmt,
)

from ...error.viserr import *
from ...error.chainerr import *
from ...catena_core.node.base import Node, NodeBus
from ...catena_core.alias.builtin import NodeType
from ...settings import RTConfig, StyleSettings, settings 

# TODO: 之后要都换成 Catenaconf
from catenaconf import Catenaconf

STYLE_CONFIG = Catenaconf.create({
    "chain_start": sty.BG,
    "chain_end": sty.BGR,
    "chain_spacer": "▬",
    "spacer_repeat": 50,
    "completion_mark": "-"
})

StyleSettings(CONFIG = STYLE_CONFIG)

def cli_visualize(vis_type: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal vis_type
            if vis_type is None:
                vis_type = settings.visualize.visualize_type
            debug("[cli_visualize] type:", vis_type)
            if vis_type == "tree":
                visualizer = TreeVisualizer()
            else:
                debug("[cli_visualize] Unsupported type:", vis_type)
                raise VisualizerTypeNotFoundError(f"Unsupported visualization type: {vis_type}")
            return visualizer.visualize(func)(*args, **kwargs)
        return wrapper

    # 如果没有传递参数，则直接返回装饰器函数
    if callable(vis_type):
        func = vis_type
        vis_type = None
        return decorator(func)
    
    return decorator

class ConsoleVisualizer(ABC):

    _inspector: Live = None
    _status: Status = None
    _content_container: Any = None
    _console_output_group: List = []

    @abstractmethod
    def update_content(self, node: Node, label: str = None, *states):
        """ 更新状态树 """
        pass

    @abstractmethod
    def visualize(self, func):
        """ 可视化装饰器 """
        pass

class TreeVisualizer(ConsoleVisualizer):
    
    _node_input_type = None
    _content_container: Tree = None

    @classmethod
    def update_content(self, label: str, contents: Any = None):
        # 深拷贝防止改变输入数据
        stat_cpy = copy.deepcopy(contents)
        debug("[update_content] stat_cpy:", stat_cpy)
        
        # 当输入是节点输入时
        if label == "Input":
            # 此时输入是大模型输入消息的列表
            if self._node_input_type == NodeType.MODEL:
                for s in stat_cpy:
                    if isinstance(s, dict):
                        if "image" in s:  # 这里是防止 base64 编码过长而对其进行截断
                            s["image"] = "base64_placeholder"
                            break        
            else:   # TODO:处理其他类型的情况
                if isinstance(stat_cpy, dict):
                    if "image" in stat_cpy:
                        stat_cpy["image"] = "base64_placeholder"
                stat_cpy = [str(stat_cpy)]
                
            # 添加输入数据到状态树    
            leaf_type = fmt.fc("Input" + ":", sty.B)
            input_leaf = self._content_container.add(leaf_type)
            for s in stat_cpy:
                input_leaf.add(s)
        elif label == "Output": 
            if not isinstance(stat_cpy, NodeBus):
                raise NodeOutputTypeError()
            completion = stat_cpy # 获取 NodeBus 对象
            
            # 对节点输出进行分条件处理
            if completion.type == NodeType.MODEL: # 输出类型是大模型输入消息的列表
                for msg in completion.main: # 此时 main 是一个列表     
                    if msg["role"] == "user" and isinstance(msg["content"], list):  # 包含图像信息
                        for content in msg["content"]:
                            if "image_url" in content:
                                url = content["image_url"]["url"]
                                truncated_url = url[:30] + "..." + url[-10:]    # 截断图像base64编码
                                content["image_url"]["url"] = truncated_url
            elif completion.type == NodeType.MODEL:  # 输出类型是大模型输出消息的字典
                completion.main = completion.main["content"]  # 只保留内容部分
            else:
                pass #TODO:更多自定义逻辑
            
            # OUTPUT 树节点
            leaf_type = fmt.fc("Output" + ":", sty.B)
            # NodeBus 对象转为列表，为树的分支
            stat_cpy = [s for s in completion.list]
            # 添加输出树节点到状态树
            oytput_leaf = self._content_container.add(leaf_type)
            for s in stat_cpy:
                oytput_leaf.add(s)    # 添加分支
        elif label == "NodeCallback":
            # CallBack 树节点
            if not isinstance(stat_cpy, NodeBus):
                raise NodeOutputTypeError()
            completion = stat_cpy # 获取 NodeBus 对象
            if completion.callback:
                # 添加回调函数节点到状态树
                leaf_type = fmt.fc("NodeCallback:", sty.B)
                completion.callback.main_input = "Output.main"
                states_callback = [s for s in completion.callback.list]
                callback_leaf = self._content_container.add(leaf_type)
                for s in states_callback:
                    callback_leaf.add(s)
        elif label == "Stop":
            self._console_output_group.pop(0)

        self._inspector.update(Group(*self._console_output_group))

    @classmethod
    def visualize(self, func):
        """ 装饰器函数，用于增强运行时的调试和可视化功能。 """
        @wraps(func)
        def wrapper(
            node: Node, input: Union[str, dict], config: RTConfig = None, *args, **kwargs
        ):
            """
            装饰器的包装函数，负责实际的调试和可视化逻辑。
            
            参数:
            - node: Node类的实例。
            - input: 运行时输入，可以是字符串或字典。
            - config: 运行时配置，用于控制行为。
            - *args, **kwargs: 其他传递给被装饰函数的参数。
            
            返回:
            - result: 被装饰函数的执行结果。
            """
            # 获取配置
            config = config or RTConfig()
            # 获取输入
            if not input:
                if not hasattr(node, "input"):
                    raise NodeInputNotFoundError()
                else:
                    input = node.input
            # 检查是否禁用了可视化调试        
            if config().get("chain_vis", False) or not settings.log.enable_chain_visualize:
                debug("[TreeVisualizer.visualize] 可视化输出关闭关闭")
                result = func(node, input, config, *args, **kwargs)
            else:
                try:
                    debug("[TreeVisualizer.visualize] node:", node.type)
                    
                    # 创建一个 Status 对象，用于显示加载动画
                    status = Status("Processing...", spinner="aesthetic", speed=2.0, refresh_per_second=10)
                    #self._inspector.update(status)
                    self._console_output_group.append(status)
                    # 创建一个 Tree 以及对象，用于显示状态信息
                    if node.latter_node:
                        thisnode = fmt.fc(node.signature, node.style)
                        nextnode = fmt.fc(node.latter_node, sty.GR_L)
                        root = f"[{thisnode} -> {nextnode}]"
                    else:
                        debug("[TreeVisualizer.visualize] node_style:", node.style)
                        root = f"[{fmt.fc(node.signature, node.style)}]"
                    self._content_container = Tree(root)
                    self._console_output_group.append(self._content_container)
                    
                    # 启动Live调试工具
                    self._inspector = Live(Group(*self._console_output_group), console=Console(), auto_refresh=True)
                    self._inspector.start()  # 开始 Live
                    self._inspector.update(Group(*self._console_output_group))
                    # 更新状态显示输入数据
                    debug("[TreeVisualizer.visualize] display_input:", node.display_input)
                    self._node_input_type = node.input_type
                    if node.display_input:
                        self.update_content("Input", input)
                    else:
                        self.update_content("Input", node.former_node + ".Output.main") 
                    # 执行被装饰的函数
                    result: NodeBus = func(node, input, config, *args, **kwargs)
                    # 更新状态显示输出结果
                    self.update_content("Output", result)
                    self.update_content("NodeCallback", result)
                    # 无论成功与否，确保停止Live调试工具
                    self.update_content("Stop")
                    self._inspector.stop()  # 停止 Live
                finally:
                    # 无论成功与否，确保停止Live调试工具
                    self.update_content("Stop")
                    self._inspector.stop()  # 停止 Live
            return result
        return wrapper
    
if __name__ == "__main__":
    pass