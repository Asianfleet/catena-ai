import traceback

class AgentError(Exception):
    """ agent 模块的异常类定义 """
    
    _info:str = None
    
    def __init__(self, message=None, code=None):
        super().__init__(message or self._info)   # 调用父类的构造函数
        self.code = code            # 错误码指定

    def __str__(self):
        if self.code:
            return f"{self.args[0]} (错误代码: {self.code})"
        return self.args[0]
    
    def trace(self):
        """打印异常的堆栈信息。"""
        traceback.print_exc()
        

class SettingsError(AgentError):
    _info = "配置中没有该项"

if __name__ == "__main__":
    try:
        raise AgentError("这是一个错误示例", 500)
    except AgentError as e:
        print(e)
        e.trace()