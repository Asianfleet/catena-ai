import re
import os
from enum import Enum
from catenaconf import Catenaconf, KvConfig
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Union
)
from pydantic import (
    BaseModel, 
    ConfigDict, 
    ValidationError, 
    Field, model_validator, create_model
)

from ..message import Message, MessageBus, MessageRole
#from ...retriever import Retriever, InputRetrieve
from ...catena_core.paths import Template
from ...catena_core.node.base import Node, NodeBus, NodeCompletion
from ...catena_core.alias.builtin import (
    NodeType as Ntype,
    PromptTemplateType as PType
)
from ...settings import info, RTConfig as RT
from ...catenasmith.cli_tools import (
    Style as sty,
    debug, info
)
from ...catenasmith.visualize.cli_visualize import cli_visualize


class SimpleInputArgs(BaseModel):
    """ 大模型提示词模板创建参数规范 """
    task: Optional[str] = Field(default=None, description="模型调用请求")
    context: Optional[str] = Field(default=None, description="检索结果或上下文信息")
    retrieve: Optional[dict] = Field(default=None, description="检索参数")
    image: Optional[Union[str, List]] = Field(default=None, description="图片")

    @model_validator(mode='before')
    def check_fields(cls, values):
        task = values.get('task')
        context = values.get('context')
        retrieve = values.get('retrieve')
        
        if task is None:
            # 如果 task 不存在，context 和 retrieve 也不能存在
            if context is not None or retrieve is not None:
                raise ValueError("context and retrieve cannot be provided if task is not set.")
        else:
            # 如果 task 存在，context 和 retrieve 只能有一个
            if context is not None and retrieve is not None:
                raise ValueError("Only one of context or retrieve can be provided when task is set.")
        return values
    
class ModelPrompt(Node):
    """
    用于生成和管理不同类型的“提示词”（prompt）的工具类，支持通过模板或直接输入来生成一系列的消息（messages）。
    可灵活处理基于字符串或模板的输入，并与大模型（如 GPT 或其他生成模型）交互。其主要功能包括：
    
    ### 模板支持：
    - from_template 类方法：根据模板字符串（或模板 ID）创建 ModelPrompt 实例，并解析模板以生成结构化的提示信息。
      支持内建模板和自定义模板，且可以处理带有视觉信息（如图片）的模板。
    - 数据验证（generate_schema 类方法）：根据输入的数据（字典或列表）动态生成 JSON 格式的 schema，用于验证输入数据的合法性。支持对复杂嵌套数据结构进行递归检查。
    - 消息生成（invoke 方法）：根据输入（字符串或字典）生成提示信息。如果使用内建模板，则会根据模板参数填充输入内容并生成相应的消息。如果使用自定义模板或无模板输入，则直接生成用户输入的消息。支持在模板中插入动态数据，并处理图像和上下文信息。
    
    ### 模板解析：
    - 支持通过正则表达式解析模板中的占位符（如 {param}），并根据实际输入动态替换占位符为对应的参数格式。
      对于包含图像等视觉信息的模板，支持特殊的处理方式。
    
    ### 支持多种输入格式：
    - 输入可以是字符串（简单文本），也可以是字典（包含任务和参数）。对于字典类型的输入，类会自动验证并处理相应的任务和检索（retriever）。
      消息管理：
    - MessageRole 类用于管理不同角色的消息（如用户消息、系统消息、上下文消息等），并支持根据输入生成相应的消息对象。
      扩展性和灵活性：
    - 支持动态创建模板模型，灵活处理模板输入，并根据具体情况选择不同的处理方式。
    """
    
    # 提示词输入
    prompt_input: Optional[str] = None
    # 节点 ID
    node_id: Ntype = Field(default=Ntype.MODEL, init=False)
    # 消息列表
    message: MessageBus = Field(default=None, init=False)
    # 模板配置 
    template: KvConfig = Field(default=None, init=False)
    # 模板类型
    template_type: PType = Field(default=PType.UD, init=False)
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
    
    @property
    def node_id(self) -> Ntype:
        return self.node_id
    
    @classmethod
    def load_prompt(cls, label: str, resolve: bool = False) -> Union[KvConfig, None]:
        if len(label.split(".")) == 2:
            file = label.split(".")[0]
            label = label.split(".")[1]
            config_path = os.path.join(Template.PROMPT_PATH, file + ".yaml")
            prompt = Catenaconf.load(config_path)[label]
            if resolve: 
                Catenaconf.resolve(prompt)
            return prompt
        else:
            return None
    
    @classmethod
    def from_template(cls, template: str, *args, **kwargs):
        """ 从模板配置中创建 PromptTemplate 实例 """
        # 1、创建实例，加载模板
        instance = cls()  
        instance.template = cls.load_prompt(template)
        # 2、判断模板类型并做出相应处理
        info("[ModelPrompt.from_template] instance.template:", instance.template)
        if instance.template:
            info("[ModelPrompt.from_template] 使用 yaml 模板")
            instance.template_type = PType.YAML
        else:
            info("[ModelPrompt.from_template] 使用字符串模板")
            
            """ 
            ########################### 从字符串新建提示词模板核心逻辑 ##############################
             1、使用正则表达式解析模板中的占位符，获取模板中的参数列表
             2、根据参数列表和模板内容生成 Catenaconf 格式的模板对象
             3、根据模板对象的 meta 字段中的 retriver_call 和 parse_call 字段，调用对应的函数
             4、根据模板对象的 message 字段，生成对应的消息列表
             5、格式：
               ```   
               角色定义以及任务指定（你是一个...助手，需要按要求完成...任务...）
               参数字段1描述：{字段名1}
               参数字段2描述：{字段名2}
               ... 
               参数字段n描述：{字段名n}
               ``` 
            ######################################################################################
            """
        
            def to_cref(match_obj: re.Match):
                """ 将模板中的占位符转换为 Catenaconf 引用 """
                content = match_obj.group(0)[1:-1]  # 获取 {} 内的内容，去掉 {} 
                if content != "image":  # 对于非图片参数，添加 Catenaconf 引用
                    return "@{" + "meta.param." + content + "}"  # 添加前缀后用 @{} 包裹返回
                else:
                    return "已经给出"
                
            # TODO: 目前该模式存在的问题：
            # 1、如果用户想要指定图片上下文，但是没有明确使用 {image} 进行标记，则不能做到优雅地
            #    切换对应的大模型提示词格式。 目前打算添加通过大模型先进行格式整理，与推测
            # 2、目前针对内建模板的 retriver_call 以及 schema、parse_call 字段还不能灵活指定
            # JSON 模式
            
            args: List = re.findall(r'\{(.*?)\}', template)   # 提取字符串中的变量占位符
            templ_ref: str = re.sub(r'\{.*?\}', to_cref, template)  # 将占位符替换为 Catenaconf 引用
            instance.template_type = PType.STR
           
            instance.template = Catenaconf.create({# 创建 Catenaconf 格式的模板对象
                "system": templ_ref,
                "meta": {
                    "retriver_call": None,
                    "model_args": None,
                    "param": {arg: None for arg in args},
                    "output": {
                        "type": "text",
                        "schema": None,
                        "parse_call": None
                    }
                }
            })

        return instance

    @classmethod
    def generate_schema(cls, obj):
        """ 给定一个字典数据，生成对应的 JSON 格式的 schema，用于数据验证 """
        if isinstance(obj, dict):
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            for key, value in obj.items():
                schema["properties"][key] = cls.generate_schema(value)
                # 如果值不为 None，则将其添加到 required 中
                if value is not None:
                    schema["required"].append(key)
        elif isinstance(obj, list):
            schema = {
                "type": "array",
                "items": cls.generate_schema(obj[0]) if obj else {}
            }
        elif isinstance(obj, str):
            schema = {"type": "string"}
        elif isinstance(obj, int):
            schema = {"type": "integer"}
        elif isinstance(obj, float):
            schema = {"type": "number"}
        elif isinstance(obj, bool):
            schema = {"type": "boolean"}
        else:
            schema = {"type": "null"}
        
        return schema

    
    def _simple_invoke(self, prompt_input: Union[str, Dict]) -> MessageBus:
        """ 
        无模板模式，此时输入应为两种情况：
        1. 字符串，直接作为用户输入；
        2. 字典，包含 task 和 retrieve 两个字段，其中 retrieve 为可选参数。
        """
        prompt_input = prompt_input or self.prompt_input
        if isinstance(prompt_input, str):
            message = MessageBus(
                [MessageRole.mask(), MessageRole.user(content=prompt_input)]
            )
            return message
        elif isinstance(prompt_input, dict):
            message = MessageBus([MessageRole.mask()])
            # 验证字典数据
            try:
                prompt_input = SimpleInputArgs(**prompt_input)  # 使用字典解包进行验证
            except ValidationError:
                raise
            #if prompt_input["retrieve"]:
            #    try:
            #        retrieve = InputRetrieve(**prompt_input["retrieve"])  # 使用字典解包进行验证
            #    except ValidationError as e:
            #        print(e.json())
            #    
            #    with Retriever(retrieve["setter"]) as retrv:
            #        retrieved = retrv.task(prompt_input["task"], **retrieve["args"])
                    
            #self.message.append(MessageRole.context(retrieved))
            message.append(MessageRole.user(prompt_input["task"]))
            
            return message

    """ def _yaml_templ_invoke(self, prompt_input: Union[str, Dict]) -> List:
        #
        #有模板模式（从 yaml 文件中读取模板），此时输入应为两种情况：
        #1. 字符串，直接作为用户输入，此时内建模板必须只包含一个参数。
        #2. 字典，包含内建模板的所有参数，其中键为参数名称，值为参数的值。
        #
        mapping = {key: (str, value) for key, value in self.template.meta.param.items()}
        TempateInputPrompt = create_model(
            "TempateInputPrompt",
            **mapping,
            __base__=BaseModel
        )

        if isinstance(prompt_input, dict):
            try:
                prompt_input = TempateInputPrompt(**prompt_input)
                prompt_input = prompt_input.model_dump()
            except ValidationError as e:
                print(e.json())

            for arg, value in prompt_input.items():
                self._template.meta.param[arg] = value
        else:
            assert len(self._template.meta.param) == 1, ""
            key = list(unwarp(self._template.meta.param).keys())
            self._template.meta.param[key[0]] = prompt_input

        if self._template.meta.retriver_call:
            pass
        if self._template.meta.model_args == None:
            self._template.meta.model_args = {}

        interpolate(self._template)
        self.message = unwarp(self._template.message)


        return self.message """

    def _str_templ_invoke(self, prompt_input) -> MessageBus:
        """ 
        有模板模式（使用字符串动态创建模板），此时输入应为两种情况：
        1. 字符串，直接作为模板字符串；
        2. 字典，包含模板字符串和模板参数。 
        """
        # 从模板中获取参数
        mapping = {
            key: (Union[str, List], value) for key, value in self.template.meta.param.items()
        }
        # 创建模板参数数据验证模型
        TempateInputPrompt = create_model(
            "TempateInputPrompt",
            **mapping,
            __base__=BaseModel
        )
        system_message: Message = MessageRole.system(content="")
        if isinstance(prompt_input, dict):
            # 验证数据并生成参数字典
            try:
                prompt_input: BaseModel = TempateInputPrompt(**prompt_input)
                args_dict: Dict = prompt_input.model_dump()  
            except ValidationError:
                raise  
            for arg, value in args_dict.items():        # 遍历传入参数
                if isinstance(value, List):
                    self.template.meta.param[arg] = "已给出"
                    system_message.images = value
                elif isinstance(value, str):
                    # 判断是否是base64或url
                    if value.startswith(('http://', 'https://')):
                        self.template.meta.param[arg] = "已给出"
                        system_message.images = value
                    else:
                        self.template.meta.param[arg] = value
        # 此时模板必须只有一个参数
        else:   
            if len(self.template.meta.param) != 1:
                raise ValueError("模板参数数量错误")
            if isinstance(prompt_input, List):
                key = list(self.template.meta.param.keys())
                self.template.meta.param[key[0]] = prompt_input
            elif isinstance(prompt_input, str):
                    # 判断是否是base64或url
                    if prompt_input.startswith(('http://', 'https://')):
                        self.template.meta.param[arg] = "已给出"
                        system_message.images = prompt_input
                    else:
                        self.template.meta.param[arg] = prompt_input
        
        #TODO:RAG支持还未实现
        if self.template.meta.retriver_call:
            pass

        Catenaconf.resolve(self.template)
        system_message.content = self.template.system

        return MessageBus([system_message])

    #@cli_visualize
    def operate(self, input: Union[NodeBus, Dict, str]) -> NodeCompletion:
        """  """
        
        if not isinstance(input, NodeBus):
            bus = NodeBus()
            bus.add(main_data=input)
        else:
            bus = input
        
        input_data : NodeCompletion = bus.latest
        prompt_input = input_data.main_data
        config = input_data.extra_data
        
        # 根据模板类型调用不同的处理函数
        if self.template_type == PType.UD:
            output_meta = {"output": {"type": None}}
            output_messages = self._simple_invoke(prompt_input)
        else:
            if self.template_type == PType.STR:
                output_messages = self._str_templ_invoke(prompt_input)
            #elif self.template_type == PType.YAML:
            #    output_messages = self._yaml_templ_invoke(prompt_input)
            output_meta = {
                "output": self.template.meta.output, 
                "model_args": self.template.meta.model_args
            }
        self.message = output_messages
        config._merge(output_meta)
        debug("[ModelPrompt] output_messages:", output_messages)
        
        bus.add(
            type=Ntype.PRM,
            main_data=output_messages,
            extra_data=config
        )
        
        return bus.latest
            
class PromptBuiltIn(Node):   
    
    pass


if __name__ == '__main__':
    # python -m catena.llmchain.prompt.prompt
    # settings.configure(disable_visualize=True)
    str_tem = """
    下面你要完成一个任务，按照给定的要求，并且参照给定的图片。
    要求：{request}
    图片：{image}
    """

    prompt = ModelPrompt.from_template(str_tem)
    
    prompt_input = {"request": "gushigushigushi", "image":["1", "2", "3"]}
    completion = prompt.operate(prompt_input)

    #print(completion.main_data.latest.to_model_message())
    
    pm = ModelPrompt(prompt_input="你好")
    msg = pm.operate("你好").main_data
    print(msg)
    print(msg.model_message)