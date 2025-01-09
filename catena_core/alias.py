from enum import Enum

class BuiltInType(Enum):
    """ 内置节点类型 """
    # 常量定义
    
    UD = "undefined"
    LLMIP = "llm-input"
    LLMOP = "llm-output"
    PAOP = "parsed-output"
    WRAPPED = "wrapped-output"   
    
class BuiltInID(Enum):
    """ 内置节点 ID  """
    
    UD = "undefined"
    LM = "llm"
    PM = "prompt"
    PS = "parser"
    MO = "memory"
    WP = "wrapper"