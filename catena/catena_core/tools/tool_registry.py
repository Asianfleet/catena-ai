from typing import Any, Dict, Union

from ...cli.tools import debug

class ToolRegistry:
    """ 工具注册类 """
    tool_regedit: Dict[str, Any] = {}
    
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
        if name not in cls.tool_regedit.keys():
            raise ValueError(f"Tool with name '{name}' not registered.")
        return cls.tool_regedit[name]