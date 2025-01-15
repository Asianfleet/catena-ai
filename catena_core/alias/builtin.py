from enum import Enum

class ModelProvider(Enum):
    """ 模型 SDK 的枚举类 """
    Qwen = "qwen"
    Kimi = "kimi"
    OpenAI = "openai"
    QianFan = "qianfan"
    DeepSeek = "deepseek"
    Anthropic = "anthropic"
    OpenAI_Compatible = "openai-compatible"

class NodeCompletionType(Enum):
    """ 内置节点类型 """
    # 常量定义
    
    UD = "undefined"
    LLMIP = "llm-input"
    LLMOP = "llm-output"
    PAOP = "parsed-output"
    WRAPPED = "wrapped-output"   
    
class NodeType(Enum):
    """ 内置节点 ID  """
    
    UD = "undefined"
    LM = "llm"
    PM = "prompt"
    PS = "parser"
    MOM = "memory"
    WP = "wrapper"
    
class MemoryType(Enum):
    """ 内置内存类型 """
    
    INF = "undefined"
    WINDOW = "window"
    TOPIC = "topic"
    
CURRENT_VERSION = "0.0.1"