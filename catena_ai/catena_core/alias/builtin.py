from enum import Enum

class ModelProvider(Enum):
    """ 模型 SDK 的枚举类 """
    Qwen = "qwen"
    Kimi = "kimi"
    OpenAI = "openai"
    QianFan = "qianfan"
    DeepSeek = "deepseek"
    Anthropic = "anthropic"
    OpenAIC = "openai-compatible"

class NodeType(Enum):
    """ 内置节点 ID  """
    
    UDFN = "undefined"
    MODEL = "model"
    PRM = "prompt"
    PARSER = "parser"
    MEM = "memory"
    WRAPPED = "wrapped"
    
class MemoryType(Enum):
    """ 内置内存类型 """
    
    INF = "undefined"
    WINDOW = "window"
    TOPIC = "topic"
    
class PromptTemplateType(Enum):
    """ 内置模板类型 """
    
    YAML = "yaml"
    STRING = "str"
    STRV = "str-vision"
    UD = "undefined"
    
CURRENT_VERSION = "0.0.1"