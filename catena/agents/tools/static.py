""" 
CatenaAI 智能体框架的工具基本接口，提供最基本的工具定义：
- 工具返回结果的统一规范 ToolCompletion
- 工具基类 BaseTool
    - 工具元数据
    - 工具执行方法
    - 工具验证方法
"""
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import (
    Any, 
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar
)

from ...error import toolserr
from ...catena_core.tools.tool import Tool
from ...cli.tools import warning, info
from ...catena_core.utils.timer import record_time

T = TypeVar("T")

class ToolCompletion(BaseModel):
    """ 工具执行结果的统一规范类 """
    output: Any = Field(default=None, description="工具执行结果")
    status: str = Field(default="success", description="执行状态")
    time_elapsed: float = Field(default=0, description="工具执行时间")
    error: Optional[str] = Field(default=None, description="错误信息")
    executed_at: datetime = Field(default_factory=datetime.now, description="执行时间戳")


@dataclass
class StaticTool(Tool, ABC):
    """ 工具配置类，用于定义工具的元数据 """
    
    name: str = field(
        default=None, 
        init=False,
        metadata={"description": "工具名称"}
    )
    description: str = field(
        default=None, 
        init=False,
        metadata={"description": "工具描述性文本"}
    )
    allow_variadic_args: bool = field(
        default=False, 
        init=False,
        metadata={"description": "是否允许变长参数"}
    )
    strict_check: bool = field(
        default=True, 
        init=False,
        metadata={"description": "是否严格检查输入参数"}
    )
    override_meta: bool = field(
        default=True, 
        init=False,
        metadata={"description": "是否覆盖函数的文档字符串与名称"}
    )
    auto_impl: bool = field(
        default=False, 
        init=False,
        metadata={"description": "是否开启自动创建功能"}
    )
    show_result: bool = field(
        default=False, 
        init=False,
        metadata={"description": "是否显示执行结果"}
    )
    stop_after_call: bool = field(
        default=False, 
        init=False,
        metadata={"description": "是否在调用后停止"}
    )
    
    def __post_init__(self):
        self.func = self._execute
    
    @property
    def func_name(self) -> str:
        """ 获取函数名称 """
        if self.name:
            return self.name
        elif self.class_name:
            return self.class_name.lower()
        raise toolserr.ToolMetaDataInitializeError("函数名称不能为空")
    
    # 子类必须实现的方法
    @abstractmethod
    @record_time
    def _execute(self) -> Type[T]:
        """ 执行工具的核心方法的抽象接口 """       
    
    @classmethod
    def execute(cls, *args, **kwargs):
        instance = cls()
        instance.generate_metadata()
        print(instance.metadata)
        instance.validate_input(*args, **kwargs)
        result = instance(*args, **kwargs)
        instance.validate_output(result)
        return result
    @abstractmethod
    def validate_input(self, *args, **kwargs):
        """ 验证输入参数 """
    
    @abstractmethod
    def validate_output(self, output: Any):
        """ 验证输出结果 """

if __name__ == "__main__":
    # python -m catena.agents.tools.static
    
    class TestTool(StaticTool):
        """ 工具测试类 """

        name: str = "test_tool"
        description: str = "这是一个测试工具"
        allow_variadic_args: bool = False
        strict_check: bool = True
        override_meta: bool = False
        auto_impl: bool = True
        show_result: bool = False
        stop_after_call: bool = False
        
        def _execute(self, input: str, repeat: int = 2) -> str:
            """ 将输入字符串的第一个字母按照repeat次重复后输出 """
        
        def validate_input(self, *args, **kwargs):
            """ 验证输入参数 """
            pass
        
        def validate_output(self, output: Any):
            """ 验证输出结果 """
            pass
        
    print(TestTool.execute("hello", repeat=3))
