from .base import BaseERR

class VisualizerTypeNotFoundError(BaseERR):
    """ 智能体结果可视化错误 """
    
if __name__ == "__main__":
    raise VisualizerTypeNotFoundError()