import traceback

class BaseERR(Exception):
    """ agent 模块的基本异常类定义 """

    _code: int = None
    
    def __init__(
        self, message: str = None, code: int = None
    ):
        # 错误类型前缀
        self.prefix = self.__class__.__name__ + ": "
        # 错误信息 
        self.message = message or self.__doc__ or "an error occurred." 
        self._code = code or self._code      # 错误码指定
        super().__init__(self.message)      # 调用父类的构造函数

    @property
    def code(self):
        """ 错误码 """
        return self._code

    # TODO: 这里的前缀有问题，和错误本身重复了
    def __str__(self):
        if self.code:
            return f"{self.prefix}{self.args[0]} (错误代码: {self.code})"
        return self.prefix + self.args[0]
    
    def printe(self):
        """打印异常信息。"""
        print(self)
    
    def trace(self):
        """打印异常的堆栈信息。"""
        traceback.print_exc()