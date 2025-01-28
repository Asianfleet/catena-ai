from abc import ABCMeta
from pydantic import BaseModel

class NodeMeta(BaseModel.__class__):
    """ Node 的元类，用于实现类之间（不是类的实例）的运算符重载和链式调用 """
    def __rshift__(cls, other):
 
        return cls() >> other
    
    def __rrshift__(cls, other):

        return other >> cls()