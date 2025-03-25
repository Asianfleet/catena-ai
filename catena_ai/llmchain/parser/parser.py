import re
import json
import traceback
from pydantic import Field
from jsonschema import validate
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Union, 
    List,
    Optional,
    TYPE_CHECKING
)

from ...catena_core.callback.node_callback import NodeCallback
from ...settings import RTConfig, info
from ...catena_core.node.base import Node, NodeCompletion
from ...catena_core.alias.builtin import NodeType
from ...catena_core.utils.format_json import delete_json_format
from ...catenasmith.visualize.cli_visualize import cli_visualize

if TYPE_CHECKING:
    from ...llmchain.response import ModelResponse
    from ...llmchain.message import Message

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
    def check(self, input: str, pattern: str) -> Union[int, str]:
        """ 检查输入字符串是否与给定的正则模式匹配 """
        if re.fullmatch(pattern, input):
            return input
        else:
            return -1

class JsonOutputChecker(BaseOutputChecker):
    """ JsonOutputChecker类，用于检查 Json 格式的大模型输出 """

    @classmethod
    def check(cls, parser_input: str, **kwargs) -> Union[Dict, int]:
        """
        检查输入是否为有效的 Json 格式，并是否符合指定的模式。
        
        Parameters:
            input: 输入的字符串，可以是 Json 格式字符串或字典。
            schema: 一个字典，用于定义 Json 模式的结构。
        
        Returns:
            如果输入符合指定的模式，返回解析后的字典；否则返回 -1。 
        """
        
        def extract_json_content(text):
            # 尝试匹配 ```json 和 ``` 之间的内容
            pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(pattern, text)
            
            if match:
                # 如果匹配到了代码块，返回代码块内的内容
                return match.group(1).strip()
            else:
                # 如果没有匹配到代码块，假设整个字符串就是 JSON
                return text.strip()
        
        try:
            input_no_code_block = extract_json_content(parser_input)
            input_dict = json.loads(input_no_code_block)
            if not kwargs.get("schema"):
                return input_dict
            validate(instance=input_dict, schema=kwargs.get("schema"))
            return input_dict
        except Exception:
            traceback_str = traceback.format_exc()  # 获取详细的错误信息
            print(traceback_str)
            return -1
  
class CodeBlockOutputChecker(BaseOutputChecker):
    """ CodeBlockOutputChecker类，用于检查代码块格式 """

    @classmethod
    def check(cls, input: str) -> Union[List[str],str]:
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
            code_content = match.group(1) if match else -1
         
            return code_content
        
        return single_parse(input)
       
            
class LLMOutputParser(Node):
    """ 大模型输出解析模块 """

    node_id: NodeType = Field(default=NodeType.PARSER, init=False)
    data_type: Optional[str] = None
    
    def check(self, input: str, type: str = None, **kwargs):
        """
        对大模型输出进行格式检查

        Parameters:
            input: 输入数据，可以是字符串或字典。
            config: 运行时配置，用于控制解析过程的行为。
        Return: 解析后的结果
        """

        content = input 
        if not type:
            if not self.data_type:
                return content
            
        type = self.data_type
        
        if type == "text":
            return StrOutputChecker()(content, kwargs.get("pattern"))
        elif type == "json_object":
            return JsonOutputChecker()(content, schema=kwargs.get("schema"))
        elif type == "codeblock":
            return CodeBlockOutputChecker()(content)
        else:
            return content
            
    def parse(self, input: Union[str, dict], config: RTConfig) -> Union[str, List]:
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
                parsed = delete_json_format(input)
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
    
    # @cli_visualize
    def operate(self, input: NodeCompletion) -> NodeCompletion:
        response: 'ModelResponse' = input.main_data
        assistant_message: 'Message' = response

        check_result = self.check(
            assistant_message.content,
            **input.extra_data.unwrap
        )
        
        print("[LLMOutputParser] check_result: ", check_result)
        
        if check_result == -1:
            cb = NodeCallback(
                source=self.type, 
                target=NodeType.LM, 
                name="regen"
            )
        else:
            cb = NodeCallback()  
        return NodeCompletion(
            type=self.node_id,
            main_data=check_result,
            extra_data=input.extra_data,
            callback=cb
        )

                
    
    
            
