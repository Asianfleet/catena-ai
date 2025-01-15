""" 
CatenaAI 智能体框架的工具基本接口，提供最基本的工具定义：
- 工具返回结果的统一规范 ToolCompletion
- 工具基类 BaseTool
    - 工具元数据
    - 工具执行方法
    - 工具验证方法
"""
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import (
    Any, 
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar
)

from catena_core.error import toolserr
from catena_core.utils.timer import record_time

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
class BaseTool(ABC, Generic[T]):
    """ 工具基类 """
    
    # 工具元数据
    name: str
    description: str
    parameters_schema: Dict
    
    @property
    def return_type(self) -> Type[T]:
        """ 工具返回结果的类型 """
        # Python 3.7 中需要直接访问 __orig_bases__
        bases = self.__class__.__orig_bases__
        if not bases:
            return Any
            
        for base in bases:
            # 在 Python 3.7 中，泛型基类会有 __origin__ 属性
            if hasattr(base, '__origin__') and base.__origin__ is BaseTool:
                if hasattr(base, '__args__'):
                    args = base.__args__
                    if args:
                        return args[0]
        return Any
    
    @abstractmethod
    def __post_init__(self):
        """ 工具实例化后初始化元数据 """
        self._initialize_metadata()
        
    @abstractmethod
    def _initialize_metadata(self) -> None:
        """初始化工具元数据"""
        pass
        
    # 子类必须实现的方法
    @abstractmethod
    @record_time
    def _execute(self, *args, **kwargs) -> Any:
        """ 执行工具的核心方法的抽象接口 """       
        pass
    
    @abstractmethod
    def validate_metadata(self) -> bool:
        """ 在执行工具之前验证元数据是否已准备完毕 """
        pass
        
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> bool:
        """ 验证输入参数 """
        return True

    @abstractmethod    
    def validate_output(self, output: Any) -> bool:
        """ 验证输出结果 """
        return True

    # 如无必要，子类可不重写此方法
    def execute(self, *args, **kwargs) -> ToolCompletion:
        """ 执行工具的核心方法 """
        
        try:
            self.validate_metadata()
            self.validate_input(*args, **kwargs)
            execute_result = self._execute(*args, **kwargs)
            self.validate_output(execute_result)
        except toolserr.ToolMetaDataValidateError as e:
            raise e
        except toolserr.ToolInputValidateError as e:
            raise e
        except toolserr.ToolOutputValidateError as e:
            raise e
        except Exception as e:
            raise e
        
        return ToolCompletion(
            output = execute_result,
            time_elapsed = self._execute.time_elapsed
        )
    
    @record_time
    def auto_create():
        """ 实验性功能，借助大模型自动生成工具 """
        pass