""" 
CatenaAI 智能体框架的工具基本接口，提供最基本的工具定义：
- 工具返回结果的统一规范 ToolCompletion
- 工具基类 BaseTool
    - 工具元数据
    - 工具执行方法
    - 工具验证方法
"""
import inspect
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import (
    Any, 
    Dict,
    Optional,
    Type,
    TypeVar
)

from catena_core.error import toolserr
from catena_core.tools.base import BaseTool
from catena_core.utils.timer import record_time
from catena_core.utils.parse_code import parse_func_object


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
class Tool(BaseTool[T]):
    """ 工具基类 """
    
    # 工具元数据
    name: str = field(
        default=None, init=True, metadata={"disciption": "工具名称"}
    )
    description: str = field(
        default=None, init=True, metadata={"disciption": "工具描述性文本"}
    )
    auto_create: bool = field(
        default=False, init=True, metadata={"disciption": "是否开启自动创建功能"}
    )
    parameters_schema: Dict = field(
        default=None, init=False, metadata={"disciption": "工具输入输出规范"}
    )
    
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
    
    def __post_init__(self):
        """ 工具实例化后初始化元数据 """
        self._initialize_metadata()
        
    def _initialize_metadata(self) -> None:
        """初始化工具元数据"""
        
        # 初始化工具名称以及描述文本
        self.name = self.name or self.__class__.__name__
        self.description = self.description or self.__doc__ or None
        # 初始化工具参数规范
        try:
            self._signature = inspect.signature(self._execute)
            self.parameters_schema = {}
            for name, param in self._signature.parameters.items():
                if name not in ('self', 'args', 'kwargs'):
                    if not param.annotation:
                        raise toolserr.ToolMetaDataInitializeError(f"工具参数 {name} 的类型注解不能为空")
                    self.parameters_schema.update({
                        name: {
                            'type': param.annotation,
                            'default': param.default if param.default is not inspect.Parameter.empty else None,
                            'required': param.default is inspect.Parameter.empty
                        }
                    })
       
            # 如果return_type未指定，从方法注解中获取
            if not self.return_type:
                raise toolserr.ToolMetaDataInitializeError("工具返回类型不能为空")
        
        except toolserr.ToolMetaDataInitializeError as e:
            raise e
    
    def auto_create():
        """ 实验性功能，借助大模型自动生成工具 """
        pass
    
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
        except toolserr.ToolOutputValidateError as e:
            raise e
        except toolserr.ToolInputValidateError as e:
            raise e
        except Exception as e:
            raise e
        
        return ToolCompletion(
            output = execute_result,
            time_elapsed = self._execute.time_elapsed
        )
    
    @record_time
    def auto_create():
        pass
    
    # 子类必须实现的方法
    @record_time
    def _execute(self, *args, **kwargs) -> Type[T]:
        """ 执行工具的核心方法的抽象接口 """       
        if self.auto_create:
            _execute_created = self.auto_create()    
            
    def validate_metadata(self) -> bool:
        """ 在执行工具之前验证元数据是否已准备完毕 """
        pass
        
    def validate_input(self, *args, **kwargs) -> bool:
        """ 验证输入参数 """
        return True

    def validate_output(self, output: Any) -> bool:
        """ 验证输出结果 """
        return True



