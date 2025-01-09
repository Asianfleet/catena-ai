from abc import ABC


class BaseModule(ABC):
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    