from typing import Dict, List, Optional, Union

from .base import BaseTool

class ToolRegistry:
    """ 工具注册类 """
    
    tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        name = tool.name
        if name in cls.tools.keys():
            raise ValueError(f"Tool with name '{name}' already registered.")
        cls.tools[name] = tool
        
    @classmethod
    def show_tools_info(cls):
        pass
    
    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        if name not in cls.tools.keys():
            raise ValueError(f"Tool with name '{name}' not registered.")
        return cls.tools[name]