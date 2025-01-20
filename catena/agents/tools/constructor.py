""" 
CatenaAI 智能体框架的工具基本接口，提供最基本的工具定义：
- 工具返回结果的统一规范 ToolCompletion
- 工具基类 BaseTool
    - 工具元数据
    - 工具执行方法
    - 工具验证方法
"""
from abc import abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import (
    Any, 
    Callable,
    Dict,
    Optional,
    Type,
    TypeVar
)

from ...error import toolserr
from ...catena_core.tools.base import BaseTool
from ...catena_core.utils.timer import record_time
from ...llmchain.message import MessageRole as MsgRole
from ...llmchain.model.minimal.llm import minimal_llm_response
from ...catena_core.utils.parse_code import parse_func_object

T = TypeVar("T")

class ToolCompletion(BaseModel):
    """ 工具执行结果的统一规范类 """
    output: Any = Field(default=None, description="工具执行结果")
    status: str = Field(default="success", description="执行状态")
    elapsed_time: float = Field(default=0, description="工具执行时间")
    error: Optional[str] = Field(default=None, description="错误信息")
    executed_at: datetime = Field(default_factory=datetime.now, description="执行时间戳")

@dataclass
class Tool(BaseTool[T]):
    """ 工具基类 """
    
    # 工具元数据
    name: str = field(
        default=None,
        metadata={"description": "工具名称"},
    )
    description: str = field(
        default=None,
        metadata={"description": "工具描述性文本"},
    )
    auto_create: bool = field(
        default=False,
        metadata={"description": "是否开启自动创建功能"},
    )
    show_result: bool = field(
        default=False,
        metadata={"description": "是否显示执行结果"},
    )
    stop_after_call: bool = field(
        default=False,
        metadata={"description": "是否在调用后停止"},
    )
    entry_function: Optional[Callable[...,Any]] = field(
        default=None,
        metadata={"description": "工具执行入口点"}
    )
    injected_by_decorator: bool = field(
        default=False,
        init=False,
        metadata={"description": "是否已通过装饰器注入工具"}
    )
    tool_schema: Dict = field(
        default_factory=dict,
        init=False,
        metadata={"description": "工具输入输出规范"},
    )
    
    @record_time
    def create_tool_auto():
        # TODO: 设置提示词语言
        llm_prompt = [
            MsgRole.system(
""" 你是一个非常智能的编程助手，十分擅长根据要求来编写代码。尤其是擅长 Python 语言。
请根据给定的 json 数据的提示，实现一个函数，并返回函数的代码。

给定的 json 数据的格式是：
    {
        'name': '函数名称', 
        'description': '对函数功能的描述', 
        'params': {
            'param1': {
                'type': param1 的参数类型（例如：<class 'int'>）, 
                'default': param1 的默认值（可以为 None）, 
                'required': param1 是否为必选参数（False 或 True）
            }
        }, 
        'return': 函数的返回值类型（例如：<class 'int'>）
    }
要求是：
    1. 函数名严格按照 name 字段
    2. 函数的功能严格按照 description 字段
    3. 函数的参数个数严格按照 params 字段（可以有多个值）
    4. 函数的参数类型、是否必选以及返回值类型严格按照 params 和 return 字段的规定
    5. 实现的函数要严格包含函数的参数、默认值（如果有）、返回值类型以及文档字符串、行间注释
    
        例如：
        def func(a: int, b: int = 1) -> int:
            '''
            函数的功能是：
            1. 输入两个整数 a 和 b
            2. 返回 a 和 b 的和
            '''
            return a + b
  
    6.  代码部分包裹在 
        ```python
            ...
        ```
        这样的格式中，并且不要输出任何多余的内容
"""
            )
        ]
        # 生成函数代码
        llm_response = minimal_llm_response(
            model="gpt-4o-mini",
            messages=llm_prompt
        )
        return llm_response
    
    # 子类必须实现的方法
    @record_time
    def _execute(self) -> Type[T]:
        """ 执行工具的核心方法的抽象接口 """       
        if self.auto_create:
            _execute_created = self.create_tool_auto()   
            print(_execute_created) 
    
    def validate_input(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现 validate_input 方法")
    
    def validate_output(self, output: Any):
        raise NotImplementedError("子类必须实现 validate_output 方法")

if __name__ == "__main__":
    # python -m catena.agents.tools.constructor

    class a(Tool[int]):
        pass
            
    def func(a: str) -> str:
        """ 将给定的字符串的第一个字符转成大写然后返回 """
                
        
    c = a(
        name = "fisrt_upper",
        description="将给定的字符串的第一个字符转成大写然后返回",
        #entry_function=func
    )
    
    t = Tool(
        name = "str_process",
        description="将给定的字符串的第一个字符重复三次然后返回",
    )
    
    print(t.tool_schema)