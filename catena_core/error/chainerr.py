from .base import BaseERR

class LLModelInitError(BaseERR):
    """ LLModel Init Error """

class ChainCompileError(BaseERR):
    """ Chain Compile Error """

class NodeOutputTypeError(BaseERR):
    """ 节点的输出应该是 NodeCompletion 类型 """
    
class NodeInputNotFoundError(BaseERR):
    """ 节点输入不存在 """