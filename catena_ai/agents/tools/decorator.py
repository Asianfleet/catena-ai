from functools import wraps
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional, 
    Union,
    overload
)

from ...cli.tools import error, debug
from ...catena_core.tools import Function, Tool


@overload
def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    allow_variadic_args: bool = False,
    strict_check: bool = False,
    override_meta: bool = False,
    auto_impl: Optional[bool] = None,
    show_result: Optional[bool] = None,
    stop_after_call: Optional[bool] = None
) -> Tool:
    ...

@overload
def tool() -> Tool:
    ...

@overload
def tool(func: Callable[..., Any]) -> Tool:
    ...

def tool(*args, **kwargs):
    """ 工具装饰器 """
    
    # 规定有效关键字
    VALID_KWARGS = frozenset({
        "name",
        "description",
        "allow_variadic_args",
        "strict_check",
        "override_meta",
        "auto_impl",
        "show_result",
        "stop_after_call"
    })
    
    # 检查是否有无效的关键字
    invalid_kwargs = set(kwargs.keys()) - VALID_KWARGS
    if invalid_kwargs:
        raise ValueError(
            f"Invalid tool configuration arguments: {invalid_kwargs}. Valid arguments are: {sorted(VALID_KWARGS)}"
        )
    
    def decorator(func):
        
        if not hasattr(func, "__is_registered__") or not func.__is_registered__:
            tool_metadata = {
                "func": func,
                **{k: v for k, v in kwargs.items() if k != "func"}
            }
            tool = Tool(**tool_metadata)    
            func.__is_registered__ = True
        
        def wrapper(*args, **kwargs):
            try:
                return tool(*args, **kwargs)
            except Exception as e:
                error(
                    f"Error in tool {tool.func_name}: {e!r}",
                    #exc_info=True,  # Include stack trace
                )
                raise
        
        wrapper.func_name = tool.func_name
        wrapper.metadata = tool.metadata
        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    
    return decorator
