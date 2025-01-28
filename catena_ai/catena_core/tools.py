from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
import inspect
import pickle
import weakref
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..error.toolserr import ToolError, CacheVersionMismatch


@dataclass
class ToolMetadata:
    """工具元数据类"""
    name: str
    description: str
    parameters_schema: Dict
    return_type: Any
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

class BaseTool(ABC):
    """工具抽象基类"""
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        self.metadata = metadata or self._generate_metadata()
        
    @abstractmethod
    def _generate_metadata(self) -> ToolMetadata:
        """生成工具元数据"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """执行工具的核心方法"""
        pass
    
    def validate_input(self, *args, **kwargs) -> bool:
        """验证输入参数"""
        # 此处可以添加具体的验证逻辑
        return True

class ToolRegistry:
    """工具注册表，用于管理所有可用的工具"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tools = {}
            cls._instance.decorated_tools = {}
        return cls._instance
    
    def register_tool(self, tool: Union[BaseTool, Callable]) -> None:
        """注册工具"""
        if isinstance(tool, BaseTool):
            self.tools[tool.metadata.name] = tool
        else:
            self.decorated_tools[tool.__name__] = tool
    
    def unregister_tool(self, tool_name: str) -> None:
        """注销工具"""
        self.tools.pop(tool_name, None)
        self.decorated_tools.pop(tool_name, None)
    
    def get_tool(self, tool_name: str) -> Optional[Union[BaseTool, Callable]]:
        """获取工具"""
        return self.tools.get(tool_name) or self.decorated_tools.get(tool_name)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: Path = Path("./tool_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = weakref.WeakKeyDictionary()
        self.cache_version = "1.0.0"
    
    def get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        params = inspect.signature(func).parameters
        # 生成参数的规范化字符串表示
        args_str = '_'.join(str(arg) for arg in args)
        kwargs_str = '_'.join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return f"{func.__name__}_{args_str}_{kwargs_str}"
    
    def save_to_cache(self, key: str, value: Any, expires_in: Optional[int] = None) -> None:
        """保存到缓存"""
        cache_data = {
            'value': value,
            'version': self.cache_version,
            'timestamp': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat() if expires_in else None
        }
        
        cache_file = self.cache_dir / f"{key}.pickle"
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_from_cache(self, key: str) -> Optional[Any]:
        """从缓存加载"""
        cache_file = self.cache_dir / f"{key}.pickle"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 检查版本
            if cache_data['version'] != self.cache_version:
                raise CacheVersionMismatch()
                
            # 检查是否过期
            if cache_data.get('expires_at'):
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if datetime.now() > expires_at:
                    return None
                    
            return cache_data['value']
        except Exception as e:
            return None

# 全局缓存管理器实例
cache_manager = CacheManager()

def tool(name: Optional[str] = None, 
         description: Optional[str] = None,
         cache_ttl: Optional[int] = None):
    """
    工具装饰器
    :param name: 工具名称
    :param description: 工具描述
    :param cache_ttl: 缓存生存时间（秒）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache_manager.get_cache_key(func, args, kwargs)
            
            # 尝试从缓存获取
            if cache_ttl is not None:
                cached_result = cache_manager.load_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 执行函数
            try:
                result = func(*args, **kwargs)
                
                # 保存到缓存
                if cache_ttl is not None:
                    cache_manager.save_to_cache(cache_key, result, cache_ttl)
                
                return result
            except Exception as e:
                raise ToolError(f"Tool execution failed: {str(e)}")
        
        # 注册工具
        tool_registry = ToolRegistry()
        tool_registry.register_tool(wrapper)
        
        return wrapper
    return decorator

# 示例使用
class Calculator(BaseTool):
    """计算器工具示例"""
    
    def _generate_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="基础的计算器工具",
            parameters_schema={
                "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            return_type=float
        )
    
    def execute(self, operation: str, x: float, y: float) -> float:
        if not self.validate_input(operation, x, y):
            raise ToolError("Invalid input parameters")
            
        operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else float('inf')
        }
        
        return operations[operation](x, y)
    
    def validate_input(self, operation: str, x: float, y: float) -> bool:
        return (
            operation in ["+", "-", "*", "/"] and
            isinstance(x, (int, float)) and
            isinstance(y, (int, float))
        )

@tool(name="string_tool", 
      description="字符串处理工具",
      cache_ttl=3600)
def string_processor(text: str, operation: str = "upper") -> str:
    """
    处理字符串的示例工具
    :param text: 输入文本
    :param operation: 操作类型 (upper/lower)
    :return: 处理后的文本
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    else:
        raise ToolError("Unsupported operation")

# 使用示例
if __name__ == "__main__":
    # 注册计算器工具
    calculator = Calculator()
    ToolRegistry().register_tool(calculator)
    
    # 使用计算器工具
    try:
        result = calculator.execute("+", 10, 20)
        print(f"计算结果: {result}")
    except ToolError as e:
        print(f"工具执行错误: {e}")
    
    # 使用字符串处理工具
    try:
        result = string_processor("Hello World")
        print(f"字符串处理结果: {result}")
    except ToolError as e:
        print(f"工具执行错误: {e}")

