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
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar
)

from ...error import toolserr
from ...catena_core.utils.timer import record_time

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
class BaseTool(Generic[T], ABC):
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
    entry_function: Optional[Callable[...,Any]] = field(
        default=None,
        metadata={"description": "工具执行入口点"}
    )
    tool_return_type: type = field(
        default=T,
        metadata={"description": "工具返回类型"}
    )
    tool_schema: Dict = field(
        default_factory=dict,
        init=False,
        metadata={"description": "工具参数的schema"}
    )
    
    @classmethod
    def __direct_subclasses__(cls) -> list:
        return cls.__subclasses__()
    
    @classmethod
    def __direct_base__(cls) -> list:
        return cls.__base__
    
    @property    
    def __class_name__(self) -> str:
        return self.__class__.__name__
        
    @property
    def return_type(self) -> type:
        """推断并返回输入类型"""
        for base in self.__class__.__orig_bases__:
            # 检查是否是泛型类型
            if hasattr(base, "__args__"):
                args = base.__args__
                if args:
                    return args[0]
        
        raise TypeError(
            f"Tool {self.__class_name__} doesn't have an inferable return_type. "
            "Override the return_type property to specify the input type."
        )
        
    @property
    def return_type_set_by_init(self) -> type:    
        return self.tool_return_type
    
    def __post_init__(self):
        """ 验证工具元数据 """
        # 1、初始化工具名称以及描述文本
        try:
            #此时类为 Tool 的子类
            if str(self.__direct_base__()) == "<class '__main__.Tool'>":    
                # 获取返回值类型
                return_type = (
                    self.return_type if self.return_type != T else self.return_type_set_by_init
                )
                # 未定义则抛出异常
                if return_type == T:
                    raise toolserr.ToolMetaDataInitializeError("未定义返回类型")
                # 名称以及描述必须都不为空
                self.name = self.name or self.__class__.__name__
                self.description = self.description or self.__doc__
                if not self.name or not self.description:
                    raise toolserr.ToolMetaDataInitializeError("工具名称和描述文本不能为空")
                # 不能指定入口函数
                if self.entry_function:
                    raise toolserr.ToolMetaDataInitializeError("此时不应指定入口函数")
            # 此时类为 Tool
            elif self.__direct_base__() == BaseTool:    
                if self.entry_function:
                    if self.injected_by_decorator:
                        raise toolserr.ToolMetaDataInitializeError("已通过装饰器注入工具")
                    else:
                        # 通过类的激活函数指定的函数，未实现时间计算逻辑，在此处添加
                        self.entry_function = record_time(self.entry_function)
                        # 名称和描述由激活函数传入决定，且不能为空
                        if not self.name or not self.description:
                            raise toolserr.ToolMetaDataInitializeError("工具名称和描述文本不能为空")
                        # 标记该函数已经被注册过 TODO：在装饰器进行检测
                        self.entry_function.__is_registered__ = True   
               
            # 2、初始化工具元参数信息
            # a. 添加工具名称以及描述
            self.tool_schema.update(
                {"name": self.name, "description": self.description, "params": {}}
            )
            # b. 获取函数签名
            # c. 更新返回类型
            if self.entry_function:    
                self._signature = inspect.signature(self.entry_function)
                self.tool_schema.update({'return': self._signature.return_annotation})
            else:
                self._signature = inspect.signature(self._execute)
                self.tool_schema.update({
                    'return': self.return_type_set_by_init 
                                if self.return_type_set_by_init != T
                                else self.return_type
                })
            # d. 获取函数输入参数
            for name, param in self._signature.parameters.items():
                # 参数名不能是args和kwargs
                if name in ('args', 'kwargs'):
                    raise toolserr.ToolMetaDataInitializeError("工具参数不能为args和kwargs")
                # 参数必须要有类型提示
                if param.annotation is inspect.Parameter.empty:
                    raise toolserr.ToolMetaDataInitializeError(f"工具参数 {name} 的类型注解不能为空")
                # 更新输入参数的数据
                self.tool_schema["params"].update({
                    name: {
                        "type": param.annotation,
                        "default": param.default if param.default is not inspect.Parameter.empty else None,
                        "required": param.default is inspect.Parameter.empty, 
                    }
                })
        except toolserr.ToolMetaDataInitializeError as e:
            raise e
        
    # 子类必须实现的方法
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> bool:
        """ 验证输入参数 """
  
    @abstractmethod
    def validate_output(self, output: Any) -> bool:
        """ 验证输出结果 """

    @abstractmethod
    @record_time
    def _execute(self, *args, **kwargs) -> Any:
        """ 执行工具的核心方法的抽象接口 """       

    # 如无必要，子类可不重写此方法
    def execute(self, **kwargs) -> ToolCompletion:
        """ 执行工具的核心方法 """
        
        try:
            self.validate_input(**kwargs)
            if self.entry_function:
                execute_result = self.entry_function(**kwargs)
                elapsed_time = self.entry_function.elapsed_time
            else:
                execute_result = self._execute(**kwargs)
                elapsed_time = self._execute.elapsed_time
            self.validate_output(execute_result)
        except toolserr.ToolOutputValidateError as e:
            raise e
        except toolserr.ToolInputValidateError as e:
            raise e
        except Exception as e:
            raise e
        
        return ToolCompletion(
            output = execute_result,
            elapsed_time = elapsed_time
        )
    
    @record_time
    def create_tool_auto():
        """ 实验性功能，借助大模型自动生成工具 """