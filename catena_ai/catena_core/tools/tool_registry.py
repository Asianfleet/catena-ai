from typing import Any, Dict, Optional, Union

from ...cli.tools import debug

class ToolRegistry:
    """ 工具注册类 """
    tool_regedit: Dict[str, Any] = {}
    tool_call_metadata: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, tool: Any):
        name = tool.func_name
        if name in cls.tool_regedit.keys():
            raise ValueError(f"Tool with name '{name}' already registered.")
        cls.tool_regedit[name] = tool
        debug(f"Tool with name '{name}' registered.")
        
    @classmethod
    def unregister(cls, tool: Union[str, Any]):
        if hasattr(tool, "func_name"):
            name = tool.func_name
        else:
            name = tool
            
        if name not in cls.tool_regedit.keys():
            raise ValueError(f"Tool with name '{name}' not registered.")
        del cls.tool_regedit[name]
        
    @classmethod
    def show_tools_info(cls):
        pass
    
    @classmethod
    def get_tool(cls, name: str) -> Any:
        """  """
        if name not in cls.tool_regedit.keys():
            raise ValueError(f"Tool with name '{name}' not registered.")
        return cls.tool_regedit[name]

    @classmethod
    def run_tool(cls, name: str, format: Optional[str] = None, *args, **kwargs) -> Union[Any, list]:
        """  """
        tool = cls.get_tool(name)
        if kwargs.get("call_id", None):
            del kwargs["call_id"]
            args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            cls.tool_call_metadata[kwargs["call_id"]] = f"{name}({args_str})"
        if format == "str":
            result = str(tool(*args, **kwargs))
        else:
            result = tool(*args, **kwargs)
        return result

