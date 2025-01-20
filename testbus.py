from catena.catena_core.node.base import Node
from catena.catena_core.node.completion import NodeBus, NodeCompletion
from catena.catena_core.alias.builtin import NodeType

class Test1(Node):
    def operate(self, input: NodeBus):
        input.update(main_data="aaaa")
    
class Test2(Node):
    def operate(self, input: NodeBus):
        input.update(main_data="bbbb")
    
    
pipe = Test1() >> Test2()
print(type(pipe.operate("1111")))