import re
import json
import traceback
from jsonschema import validate
from abc import ABC, abstractmethod
from typing import Any

from src.utils.util import Tools
from ..catena_core.settings import RTConfig, info
from ..catena_core.callbacks import Callback
from ..catena_core.nodes import Node, NodeCompletion
from ..catena_core.alias import BuiltInType, BuiltInID
from ..catenasmith.visualize.cli_visualize import cli_visualize

class BaseOutputChecker(ABC):
    """ 输出检查器基类 """

    @classmethod
    @abstractmethod
    def check(cls, parser_input: Any, *args, **kwargs):
        pass
    
    @classmethod
    def __call__(cls, parser_input: Any, *args, **kwargs):
        return cls.check(parser_input, **kwargs)

class StrOutputChecker(BaseOutputChecker):
    def check(self, input: list[str] | str) -> list[str] | str:
        return input + "_parsed"

class JsonOutputChecker(BaseOutputChecker):
    """ JsonOutputChecker类，用于检查 Json 格式的大模型输出 """

    @classmethod
    def check(cls, parser_input: str, **kwargs) -> list[str] | str:
        """
        检查输入是否为有效的 Json 格式，并是否符合指定的模式。
        
        Parameters:
            input: 输入的字符串，可以是 Json 格式字符串或字典。
            schema: 一个字典，用于定义 Json 模式的结构。
        
        Returns:
            如果输入符合指定的模式，返回解析后的字典；否则返回 -1。 
        """
        try:
            parser_input = json.loads(parser_input)
            validate(instance=parser_input, schema=kwargs.get("schema"))
            return parser_input
        except Exception:
            traceback_str = traceback.format_exc()  # 获取详细的错误信息
            print(traceback_str)
            return -1
  
class CodeBlockOutputChecker(BaseOutputChecker):
    """ CodeBlockOutputChecker类，用于检查代码块格式 """

    @classmethod
    def check(cls, input: list[str] | str) -> list[str] | str:
        """
        检查输入是否为有效的代码块格式，并返回解析后的代码内容。
        
        Parameters:
            input: 输入的字符串，可以是代码块格式字符串或字典。
        
        Returns:
            如果输入符合代码块格式，返回解析后的代码内容；否则返回 -1。
        """
        def single_parse(input):
            
            # 正则表达式，用于匹配代码块的内容
            pattern = r"```.*?\n(.*?)\n```"
            # 执行查找
            match = re.search(pattern, input, re.DOTALL)
            # 获取代码块内容
            code_content = match.group(1) if match else None
         
            return code_content
            
        if isinstance(input, str):
            return single_parse(input)
        else:
            return [single_parse(inp) for inp in input]
            
class LLMOutputParser(Node):
    """ 大模型输出解析模块 """

    id = BuiltInID.PS

    def __init__(self):
        super().__init__("bold yellow")
    
    def check(self, input: list | str, config: RTConfig = None):
        """
        对大模型输出进行格式检查

        Parameters:
            input: 输入数据，可以是字符串或字典。
            config: 运行时配置，用于控制解析过程的行为。
        Return: 解析后的结果
        """

        content = input if isinstance(input, str) else input["content"]
        if not config:
            return content
        elif config().output.type == "text":
            return StrOutputChecker()(content, config().output.re_pattern)
        elif config().output.type == "json_object":
            schema = config.unwrap
            return JsonOutputChecker()(content, schema=schema)
        elif config().output.type == "codeblock":
            return CodeBlockOutputChecker()(content)
        else:
            return content
            
    def parse(self, input: str | dict, config: RTConfig) -> list | str:
        """
        对大模型输出进行解析，并返回解析后的结果。
        
        Parameters:
            input: 输入数据，类型为字符串。
            config: 运行时配置，用于控制解析过程的行为。
        Return: 解析后的结果
        """
        
        if config.output.parse_call is not None:
            if config.output.operation.ops == "split":
                info("输出解析模式：字符串分割成列表")
                parsed = input.split(config.output.operation.operator)
            elif config.output.operation.ops == "jsondeletef":
                info("输出解析模式：去除 JSON 格式")
                parsed = Tools.delete_json_format(input)
            elif config.output.operation.ops == "jsonselect":
                info("输出解析模式：JSON 选择性输出")
                try:
                    if config.output.operation.operator:
                        operator = config.output.operation.operator
                    else:
                        operator = config.operator
                    for key in operator.split("-"):
                        parsed = input[key]
                        input = parsed
                except Exception:
                    traceback_str = traceback.format_exc()
                    print(traceback_str)
            elif config.output.operation.ops == "sheet_cvt":
                pass
        else:
            parsed = input

        return parsed
    
    @cli_visualize
    def operate(
        self, input: list | str, config: RTConfig=None, *args, **kwargs
    ):

        check_result = self.check(input, config)
        
        if check_result == -1:
            cb = Callback(
                source=self.type, 
                target=BuiltInID.LM, 
                name="regen"
            )
        else:
            cb = Callback()  
        return NodeCompletion(
            BuiltInType.PAOP, check_result, args, kwargs, config, cb
        )

                
    
    
            
