from abc import ABC, ABCMeta
from pydantic import BaseModel


""" class Meta(ABCMeta):
    pass



class BaseTool(ABC, metaclass=Meta):
    pass


class MetaTool(BaseTool.__class__, BaseModel.__class__):
    pass

class Tool(BaseModel, BaseTool, metaclass=MetaTool):
    pass
 """
 
""" class Meta(BaseModel.__class__):
    pass

class BaseTool(BaseModel, metaclass=Meta):
    pass

class Tool(ABC, BaseTool):
    pass """
    
from pydantic import BaseModel

class Meta(type(BaseModel)):
    pass

class ReadOnlyModel(BaseModel, metaclass=Meta):
    _read_only_field: int

    def __post__init__(self):
        self._read_only_field = 1

    @property
    def read_only_field(self):
        return self._read_only_field
ReadOnlyModel()