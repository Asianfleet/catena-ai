from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Dict, 
    List, 
    Union
)

from ...catenasmith.cli_tools import (
    info,
    debug,
    Style as sty
)
from ...catenasmith.visualize.cli_visualize import cli_visualize

from ...settings import RTConfig
from ...catena_core.utils.utils import MessageRole
from ...catena_core.node.base import Node, NodeBus
from ...catena_core.callback.node_callback import node_callback
from ...catena_core.alias.builtin import NodeType, MemoryType

class Memory(Node):
    pass

@dataclass
class InfMemory(Memory):
    """
    默认的记忆类，可以存储任意类型的消息，包括系统消息和用户消息。
    """
    
    node_id: Enum = field(default=NodeType.MEM, init=False, metadata={"disciption": "节点id"})
    system_prompt: List[str] = field(default=None, metadata={"disciption": "系统提示语"})
    chat_message: List[Dict] = field(default_factory=list, init=True, metadata={"disciption": "对话消息"})
    messages: List[Dict] = field(default_factory=list, init=False, metadata={"disciption": "全部消息"})
    system_message: List[Dict] = field(default_factory=list, init=False, metadata={"disciption": "系统消息"})
               
    def __post_init__(self):
        if self.system_prompt:
            self.system_message = [MessageRole.system(prompt) for prompt in self.system_prompt]
        super().__init__(sty.BC)

    @property
    def node_id(self):
        return self.node_id

    def add(self, message: Union[Dict, List[Dict]]):
        """
        将消息添加到记忆中。

        根据消息中的角色字段（"role"），将消息分类为系统消息或其他消息，并分别添加到对应的列表中。
        如果消息是字典列表，将遍历列表并对每个字典进行分类和添加。
        最后，将其他消息列表扩展到系统消息列表中，并返回扩展后的系统消息列表（注意，此操作不会修改原系统消息列表）。

        :param message: 要添加的消息，可以是单个字典或字典列表。
        :type message: Union[Dict, List[Dict]]
        :return: 扩展后的系统消息列表。
        :rtype: List[Dict]
        """
        system_message = self.system_message.copy()
        chat_message = self.chat_message.copy()

        def sortmsg(msg: Dict):
            """
            根据消息中的角色字段对消息进行分类，并添加到对应的列表中。
            """
            if msg["role"] == "system":
                if msg["content"] not in [msg["content"] for msg in system_message]:
                    system_message.append(msg)
            else:
                chat_message.append(msg)
        
        messages = [message] if isinstance(message, dict) else message
        for msg in messages:
            sortmsg(msg)
        system_message.extend(chat_message)
        # 将其他消息列表扩展到系统消息列表中，并返回
        return system_message
        
    @node_callback("update_memory")
    def add_(self, message: Union[Dict, List[Dict]]):
        """
        将传入的消息添加到对应的消息列表中，并触发更新回调。

        """
        # 定义一个内部函数，用于根据消息的角色将其添加到不同的消息列表中
        def sortmsg(msg):
            # 如果消息的角色为系统消息
            if msg["role"] == "system":
                # 将消息添加到系统消息列表中
                if msg["content"] not in [msg["content"] for msg in self.system_message]:
                    # 如果消息的内容不在系统消息列表中，则添加
                    self.system_message.append(msg)
            else:
                # 否则将消息添加到其他消息列表中
                self.chat_message.append(msg)
        messages = [message] if isinstance(message, dict) else message
        for msg in messages:
            # 遍历列表中的每个消息，调用sortmsg函数进行分类处理
            sortmsg(msg)
        self.messages = [*self.system_message, *self.chat_message]
        debug("[add_] messages:", self.messages)

    def pop(self):
        self.messages = self.messages[:-2]
    
    def clear(self):
        self.messages = []

    @cli_visualize
    def operate(self, input: List, config: RTConfig = None, *args, **kwargs):
        """ 大模型记忆类的启动函数 """
        self.add_(input)
        return NodeBus(
            NodeType.MODEL, 
            main=self.messages, 
            args=args, 
            kwargs=kwargs, 
            config=config
        )

class WindowMemory(Memory):
    pass
    
class TopicMemory(Memory):
    pass

def init_memory(memory_type: MemoryType = MemoryType.INF):

    if memory_type == MemoryType.INF:
        return InfMemory()
    elif memory_type == MemoryType.WINDOW:
        return WindowMemory()
    elif memory_type == MemoryType.TOPIC:
        return TopicMemory()
    else:
        raise ValueError("Invalid memory type: {}".format(memory_type))


if __name__ == "__main__":
    # python -m src.modules.agent.llm.memory
    print("llm memory test")
   