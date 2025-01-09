from abc import ABC, abstractmethod
from collections import deque
from typing import Union, Callable, TypeVar, Any, List, Dict, Optional
from ..core import Node
from ..catena_core.settings import RTConfig

import logging
#from src import loginit
logger = logging.getLogger(__name__)

LLMemory = TypeVar("Memory", List[Dict], Dict)

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

class InfMemory(LLMBaseMemory):
    """
    默认的记忆类，可以存储任意类型的消息，包括系统消息和用户消息。
    """

    def __init__(self, regularizer: Callable[..., Any] = None):
        self.regularizer = regularizer if regularizer is not None else lambda x: x
        self.system_prompt = []
        self.other_prompt = []
        self.messages = []
        self.structured_output = []
               
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
        system_prompt = self.system_prompt.copy()
        other_prompt = self.other_prompt.copy()

        def sortmsg(msg: Dict):
            """
            根据消息中的角色字段对消息进行分类，并添加到对应的列表中。
            """
            if msg["role"] == "system":
                system_prompt.append(msg)
            else:
                other_prompt.append(msg)
        
        # 处理字典列表消息
        if isinstance(message, list):
            for msg in message:
                sortmsg(msg)
        # 处理单个字典消息
        elif isinstance(message, dict):
            sortmsg(message)

        system_prompt.extend(other_prompt)
        # 将其他消息列表扩展到系统消息列表中，并返回
        return system_prompt
        
    def add_(self, message: Union[Dict, List[Dict]]):
        """
        将传入的消息添加到对应的消息列表中，并触发更新回调。

        """
        # 定义一个内部函数，用于根据消息的角色将其添加到不同的消息列表中
        def sortmsg(msg):
            # 如果消息的角色为系统消息
            if msg["role"] == "system":
                # 将消息添加到系统消息列表中
                    self.system_prompt.append(msg)
            else:
                # 否则将消息添加到其他消息列表中
                self.other_prompt.append(msg)
        # 判断传入的消息类型
        if isinstance(message, list):
            # 如果消息是列表类型
            for msg in message:
                # 遍历列表中的每个消息，调用sortmsg函数进行分类处理
                sortmsg(msg)
        elif isinstance(message, dict):
            # 如果消息是字典类型
            sortmsg(message)

        # 触发更新回调
        self._update_callback()

    def pop(self):
        self.messages = self.messages[:-2]
    
    def clear(self):
        self.messages = []

    def operate(self, input: LLMemory, config: RTConfig = None, *args, **kwargs):
        assert isinstance(input, list), "input must be a list of dict"
        logger.info("\n当前组件：MEMORY\n组件输入：%s", input)
        self.input = input
        if all(isinstance(inp, dict) for inp in input):
            self.messages = self.add(input)
        elif all(isinstance(inp, list) for inp in input):
            self.messages = input

        logger.info("\n组件输出：%s\n当前组件：MEMORY", self.messages)
        return self.messages

    def _callback(
        self, future_result, config: RTConfig, *args, **kwargs
    ):
        """
        回调函数，用于将当前输入添加到记忆库（如果设置了add_to_memory为True）
        
        Args:
            config (RTConfig, optional): 运行时配置信息. Defaults to None.
            *args: 可变位置参数.
            **kwargs: 可变关键字参数.
        
        Returns:
            None
        """
        logger.info("Callback: %s", self.__class__.__name__)
        if config.get("add_to_memory", True):
            logger.info("本次对话将被保存")
            self.add_(self.input)
            self.add_(future_result)
        else:
            logger.info("本次对话将被舍弃")

    def _update_callback(self):
        """
        更新回调函数，用于更新聊天内存。
        """

        messages = []
        self.messages = []
        messages.extend(self.system_prompt)
        messages.extend(self.other_prompt)
        self.messages = messages
  
class WindowMemory(LLMBaseMemory):
    def __init__(self, max_len=10):
        # 创建一个双端队列，最大长度为max_len
        self.memory = deque(maxlen=max_len)

    def add_message(self, message):
        # 添加一条新的消息到队列中
        self.memory.append(message)

    def get_memory(self):
        # 返回当前所有的记忆
        return list(self.memory)
    
class TopicMemory(LLMBaseMemory):
    def __init__(self):
        # 使用字典来存储不同话题的记忆
        self.memory = {}

    def add_message(self, topic, message):
        # 如果话题不存在，先创建一个新的列表
        if topic not in self.memory:
            self.memory[topic] = []
        # 添加消息到对应话题的列表中
        self.memory[topic].append(message)

    def get_memory_by_topic(self, topic):
        # 返回指定话题的所有记忆
        return self.memory.get(topic, [])

    def get_all_topics(self):
        # 返回所有话题的列表
        return list(self.memory.keys())

    def get_all_memories(self):
        # 返回所有话题和对应的记忆
        return self.memory

def init_memory(memory_type:str = "default"):

    if memory_type == "default":
        return InfMemory()
    elif memory_type == "window":
        return WindowMemory()
    elif memory_type == "topic":
        return TopicMemory()
    else:
        raise ValueError("Invalid memory type: {}".format(memory_type))

Memory = Union[LLMBaseMemory, str]


if __name__ == "__main__":
    # python -m src.modules.agent.llm.memory
    print("llm memory test")
   