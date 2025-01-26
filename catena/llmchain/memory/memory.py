from enum import Enum
from pydantic import Field
from typing import (
    Dict, 
    List, 
    Union
)

from ...cli.tools import (
    info,
    debug,
    Style as sty
)
from ...catenasmith.visualize.cli_visualize import cli_visualize

from ..message import Message, MessageBus, MessageRole
from ...catena_core.node.base import Node, NodeBus, NodeCompletion
from ...catena_core.callback.node_callback import node_callback
from ...catena_core.alias.builtin import NodeType, MemoryType

class Memory(Node):
    pass

class InfMemory(Memory):
    """
    默认的记忆类，可以存储任意类型的消息，包括系统消息和用户消息。
    """
    
    system_prompt: Union[List[str], str] = Field(
        default=None, metadata={"disciption": "系统提示语"}
    )
    chat_message: MessageBus = Field(
        default_factory=MessageBus, init=False, metadata={"disciption": "对话消息"}
    )
    messages: MessageBus = Field(
        default_factory=MessageBus, init=False, metadata={"disciption": "全部消息"}
    )
    system_message: MessageBus = Field(
        default_factory=MessageBus, init=False, metadata={"disciption": "系统消息"}
    )
    node_id: Enum = Field(default=NodeType.MEM, init=False, metadata={"disciption": "节点id"})
    style: Enum = Field(default=sty.BC, init=False, metadata={"disciption": "节点样式"})
               
    def model_post_init(self, __pydantic_extra__=None):
        if self.system_prompt:
            prompts = [self.system_prompt] if isinstance(self.system_prompt, str) else self.system_prompt
            self.system_message = MessageBus(
                [MessageRole.system(prompt) for prompt in prompts]
            )

    def add(self, message: Union[Message, MessageBus]):
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
        system_message = self.system_message.deepcopy()
        chat_message = self.chat_message.deepcopy()

        def sortmsg(msg: Message):
            """
            根据消息中的角色字段对消息进行分类，并添加到对应的列表中。
            """
            if msg.role == "system":
                if msg.content not in [msg.content for msg in system_message]:
                    system_message.add(msg)
            else:
                chat_message.add(msg)
        
        messages = (
            MessageBus([message]) if isinstance(message, Message) else message
        )
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
        def sortmsg(msg: Message):
            # 如果消息的角色为系统消息
            if msg.role == "system":
                # 将消息添加到系统消息列表中
                if msg.content not in [msg.content for msg in self.system_message]:
                    # 如果消息的内容不在系统消息列表中，则添加
                    self.system_message.append(msg)
            else:
                # 否则将消息添加到其他消息列表中
                self.chat_message.append(msg)
        messages = MessageBus([message]) if isinstance(message, Message) else message
        for msg in messages:
            # 遍历列表中的每个消息，调用sortmsg函数进行分类处理
            sortmsg(msg)
        self.messages = MessageBus([*self.system_message, *self.chat_message])
        debug("[add_] messages:", self.messages)

    def pop(self) -> None:
        """移除最后两条消息
        
        从messages中移除最后两条消息。如果messages中元素少于2条，
        则清空messages。
        """
        if len(self.messages) >= 2:
            self.messages = MessageBus(list(self.messages)[:-2])
        else:
            self.messages.clear()
    
    def clear(self):
        self.messages = MessageBus([])

    #@cli_visualize
    def operate(self, input: NodeCompletion) -> NodeCompletion:
        """ 大模型记忆类的启动函数 """
        prompt_messages: MessageBus = input.main_data
        self.add_(prompt_messages)
        return NodeCompletion(
            type=NodeType.MODEL, 
            main_data=self.messages
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
    # python -m catena.llmchain.memory.memory
    mem = InfMemory(system_prompt="你是一个助手")
    mem.add_(MessageRole.user("你好"))
    mem.add_(MessageRole.system("系统提示词"))
    print(mem.messages)