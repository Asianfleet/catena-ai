from .base_err import BaseERR

class ToolError(BaseERR):
    """ 智能体工具错误 """

class ToolMetaDataInitializeError(BaseERR):
    """ 工具类元数据初始化错误 """

class ToolMetaDataValidateError(BaseERR):
    """ 工具类元数据校验错误 """

class ToolInputValidateError(BaseERR):
    """ 工具输入校验错误 """
    
class ToolOutputValidateError(BaseERR):
    """ 工具输出校验错误 """

class ToolValidateError(BaseERR):
    """ 工具校验错误 """

class CacheVersionMismatch(BaseERR):
    """ 缓存版本不匹配 """
    