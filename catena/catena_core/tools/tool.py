""" 
CatenaAI 智能体框架的工具基本接口，提供最基本的工具定义：
- 工具返回结果的统一规范 ToolCompletion
- 工具基类 Tool
    - 工具元数据
    - 工具执行方法
    - 工具验证方法
"""
from datetime import datetime
from abc import abstractmethod
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import (
    Any, 
    Optional,
    TypeVar
)

from ...error import toolserr
from .function import Function
from .tool_registry import ToolRegistry as Registry
from ...catenasmith.cli_tools import debug

# 定义泛型类型变量，用于规定工具返回类型
T = TypeVar("T")

class ToolCompletion(BaseModel):
    """ 工具执行结果的统一规范类 """
    output: Any = Field(default=None, description="工具执行结果")
    status: str = Field(default="success", description="执行状态")
    time_elapsed: int = Field(default=0, description="工具执行时间")
    error: Optional[str] = Field(default=None, description="错误信息")
    executed_at: datetime = Field(default_factory=datetime.now, description="执行时间戳")

@dataclass
class Tool(Function):
    """ 工具基类 """
    
    # 工具元数据
    name: str = field(
        default=None,
        metadata={"description": "工具名称"},
    )
    description: Optional[str] = field(
        default=None,
        metadata={"description": "工具描述性文本"}
    )
    show_result: bool = field(
        default=False,
        metadata={"description": "是否显示执行结果"}
    )
    stop_after_call: bool = field(
        default=False,
        metadata={"description": "是否在调用后停止"}
    )
        
    def __post_init__(self):
        """ 注册工具 """
        debug(f"Initializing tool {self.func_name}")
        super().__post_init__()
        self.register()
        
    # 子类必须实现的方法
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> bool:
        """ 验证输入参数 """

    @abstractmethod
    def validate_output(self, output: Any) -> bool:
        """ 验证输出结果 """   

    @abstractmethod
    def execute(self, *args, **kwargs) -> ToolCompletion:
        """ 执行工具的核心方法 """
        
        try:
            self.validate_input(*args, **kwargs)
            exec_result = self.func(*args, **kwargs)
            self.validate_output(exec_result)
        except toolserr.ToolOutputValidateError as e:
            raise e
        except toolserr.ToolInputValidateError as e:
            raise e
        except Exception as e:
            raise e
        
        return ToolCompletion(
            output = exec_result,
            time_elapsed = self.func.time_elapsed,
        )
    
    def register(self) -> None:
        """ 将工具注册到工具库 """
        debug(f"Registering tool {self.func_name}")
        Registry.register(self)
        
    def unregister(self) -> None:
        """ 从工具库中注销工具 """
        Registry.unregister(self)