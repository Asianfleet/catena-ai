import inspect
from typing import Callable, Any, Dict, Type, Optional

def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    获取函数的参数及其类型注解，以及返回值的类型注解。

    :param func: 要检查的函数。
    :return: 包含参数名和类型注解的字典，以及返回值的类型注解。
    """
    signature = inspect.signature(func)
    
    parameters_schema = {
        name: {
            'type': param.annotation,
            'default': param.default if param.default is not inspect.Parameter.empty else None,
            'required': param.default is inspect.Parameter.empty
        }
        for name, param in signature.parameters.items()
        if name not in ('self', 'args', 'kwargs')
    }
    
    print("函数签名信息:", parameters_schema)

# 示例函数
def example_function(a: int, b: float, c: str = "default") -> bool:
    return a > 0 and b < 1.0 and c.startswith("d")

def ex(a) -> Any:
    return a

# 获取函数签名信息
signature_info = get_function_signature(example_function)
print(signature_info)
