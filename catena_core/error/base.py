import traceback

class BaseERR(Exception):
    """ agent 模块的基本异常类定义 """
    
    _code: int = None
    _info: str = None
    
    def __init__(self, msg: str = None, code: int = None):
        self.msg = msg or self._info or self.__doc__    # 错误信息
        self.code = code or self._code  # 错误码指定
        super().__init__(self.msg)      # 调用父类的构造函数

    def __str__(self):
        if self.code:
            return f"{self.args[0]} (错误代码: {self.code})"
        return self.args[0]
    
    def trace(self):
        """打印异常的堆栈信息。"""
        traceback.print_exc()