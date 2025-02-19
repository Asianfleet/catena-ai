import os
import re
import pickle
import inspect
from inspect import Parameter
from time import perf_counter
from dataclasses import dataclass, field
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union
)

from ...error import toolserr
from ...cli.tools import warning, info
from ...catena_core.utils.timer import record_time
from ...llmchain.message import MessageRole as MsgRole
from ...llmchain.client.minimal.llm import system_llm_response


# 将路径拆分为各个部分
base_path = os.path.dirname(os.path.abspath(__file__))
catena_ai_path = os.path.dirname(os.path.dirname(base_path))

FUNCTION_CACHE = os.path.join(catena_ai_path, ".data/.cache/functions_impl.pkl")


@dataclass
class Function:
    """ 智能体工具函数封装类 """
    
    name: Optional[str] = field(
        default=None,
        metadata={"description": "函数名称"}
    )
    description: Optional[str] = field(
        default=None,
        metadata={"description": "函数描述性文本"}
    )
    func: Optional[Callable[..., Any]] = field(
        default=None,
        metadata={"description": "函数对象"}
    )
    allow_variadic_args: Optional[bool] = field(
        default=False,
        metadata={"description": "函数是否允许变长参数"}
    )
    strict_check: Optional[bool] = field(
        default=False,
        metadata={"description": "函数是否严格检查类型"}
    )
    override_meta: Optional[bool] = field(
        default=False,
        metadata={"description": "是否覆盖函数的文档字符串与名称"}
    )
    auto_impl: Optional[bool] = field(
        default=False,
        metadata={"description": "是否自动创建函数"}
    )
    metadata: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        metadata={"description": "函数的元数据"}
    )
    
    @property
    def func_doc(self) -> str:
        """ 获取函数文档 """
        if self.func:
            if self.func.__doc__:
                if self.description and self.override_meta:
                    warning(
                        f"函数 {self.func.__name__} 的文档将被 {self.class_name}.description 覆盖", 
                        prefix="func_doc"
                    )
                    return self.description
                return self.func.__doc__
            else:
                if self.description:
                    warning(
                        f"函数文档为空，使用 {self.class_name}.description 作为函数文档", prefix="func_doc"
                    )
                    return self.description
                else:
                    raise toolserr.ToolMetaDataInitializeError("函数文档不能为空")
        else:
            warning(f"未提供函数", prefix="func_doc")
            if not self.description:
                raise toolserr.ToolMetaDataInitializeError("函数描述性文本不能为空")
            return self.description
    
    @property
    def func_name(self) -> str:
        """ 获取函数名称 """
        if self.func:
            if self.override_meta:
                warning(
                    f"函数 {self.func.__name__} 的名称将被 {self.class_name}.name 覆盖", 
                    prefix="func_name"
                )
                return self.name
            return self.func.__name__
        else:
            warning(f"未提供函数，使用 {self.class_name}.name 属性", prefix="func_name")
            if not self.name:
                raise toolserr.ToolMetaDataInitializeError("函数名称不能为空")
            return self.name
    
    @property
    def allow_variadic(self) -> bool:
        """ 获取函数是否允许变长参数 """
        return self.allow_variadic_args
       
    @property
    def require_strict_check(self) -> bool:
        """ 获取函数是否严格检查类型 """
        return self.strict_check
    
    @property
    def class_name(self) -> str:
        """ 获取函数所属的类名 """
        return self.__class__.__name__
    
    @property
    def exec_time_elapsed(self) -> float:
        """ 获取函数执行时间 """
        if self.time_elapsed:
            return self.time_elapsed
        else:
            return None
        
    @property
    def impl_time_elapsed(self) -> float:
        """ 获取函数自动创建时间 """
        if hasattr(self.auto_implement, "time_elapsed"):
            return self.auto_implement.time_elapsed
        else:
            return None
            
    def __post_init__(self):
        """ 初始化函数元数据 """
        self.generate_metadata()
        
    def parse_function_description(self, description) -> Dict[str, Any]:
        """ 解析函数描述 """
        patterns = [
            r"(.*)\[parameters\](.*)\[return\](.*)",
            r"(.*)\[parameters\](.*)",
            r"(.*)\[return\](.*)"
        ]
        
        param_pattern = re.compile(
            r'\s*-\s*([a-zA-Z]+)([（(])([^（）()]+)([）)])\s*[:：]\s*(.*?)\n'
        )
        
        return_pattern = re.compile(
            r'\s*([a-zA-Z]+)([（(])([^（）()]+)([）)])\s*[:：]\s*(.*?)'
        )
        parsed = {"overall": None, "param_desc": {}, "return_type": None}
        for pattern in patterns:
            match = re.match(pattern, description, re.DOTALL)
            if match:
                overall = match.group(1).replace("[description]", "")
                parsed["overall"] = overall.strip()
                for group in match.groups()[1:]:
                    mathes_p = param_pattern.findall(group)
                    mathes_r = return_pattern.findall(group)
                    if mathes_p:
                        parsed["param_desc"] = {
                            m[0]: m[4] for m in mathes_p
                        }
                        continue
                    if mathes_r:
                        parsed["return_type"] = mathes_r[0][2]
                        continue
                break
        if not parsed["overall"]:
            parsed["overall"] = description
        return parsed

    def generate_metadata(self):
        """ 生成函数元数据 """
        self.parsed_meta: Dict = self.parse_function_description(self.func_doc)
        if self.func:
            self.metadata.update({
                "name": self.func_name, 
                "description": self.parsed_meta["overall"]
            })
            self.signature = inspect.signature(self.func)
            self.parameters = self.signature.parameters
            if self.strict_check:
                if not self.parameters.items():
                    raise toolserr.ToolMetaDataInitializeError("不允许空参数")
                if self.signature.return_annotation == Parameter.empty:
                    raise toolserr.ToolMetaDataInitializeError("不允许空返回类型")
            self.metadata.update({"parameters": {}})
            for name, pm in self.signature.parameters.items():
                # 检查变长参数
                if not self.allow_variadic and name in ('args', 'kwargs'):
                    raise toolserr.ToolMetaDataInitializeError("不允许变长参数")
                # 检查类型提示
                if self.strict_check and pm.annotation is Parameter.empty:
                    raise toolserr.ToolMetaDataInitializeError(f"参数 {name} 需要类型注解")
                # 更新输入参数的数据
                param: Dict = {
                    name: {
                        "type": pm.annotation,
                        "default": pm.default 
                                if pm.default is not Parameter.empty 
                                else None,
                        "required": pm.default is Parameter.empty, 
                    }
                }
                self.metadata["parameters"].update(param)
            if self.parsed_meta.get("param_desc"):
                self.metadata.update({"param_desc": self.parsed_meta["param_desc"]})
                if (
                    self.metadata["parameters"].keys() 
                    != 
                    self.parsed_meta["param_desc"].keys()
                ):
                    raise toolserr.ToolMetaDataInitializeError("参数描述与函数签名不匹配")
            self.metadata.update({"return": self.signature.return_annotation})
        else:
            self.strict_check = False
            self.allow_variadic_args = False
            self.metadata.update(
                {"name": self.name, "description": self.parsed_meta["overall"]}
            )
        
    def validate_actual_work(self, func):
        """ 工具是否包含实际工作 """
        import dis
        code = func.__code__
        instructions = list(dis.get_instructions(code))
        # Check for meaningful operations
        minimum_meaningful_ops = {
            "LOAD_CONST", "STORE_FAST", "LOAD_FAST", "RETURN_VALUE"
        }
        actual_work = minimum_meaningful_ops.issubset(
            set(instr.opname for instr in instructions)
        )
    
        return actual_work
        
    @record_time
    def auto_implement(self):
        """ 自动创建函数 """
        info(f"开始自动创建函数 {self.func_name} ...")
        # TODO: 设置提示词语言
        llm_prompt = [
            MsgRole.system(
""" 
你是一个非常智能的编程助手，十分擅长根据要求来编写代码。尤其是擅长 Python 语言。
请根据给定的 json 数据的提示，实现一个函数，并返回函数的代码。

给定的 json 数据的格式是：
    {
        'name': '函数名称', 
        'description': '对函数功能的描述', 
        'params': {
            'param1': {
                'type': param1 的参数类型（例如：<class 'int'>）, 
                'default': param1 的默认值（可以为 None）, 
                'required': param1 是否为必选参数（False 或 True）
            }
        }, 
        "param_desc': {
            'param1': param1 的详细描述
        },
        'return': 函数的返回值类型（例如：<class 'int'>）
    }
要求是：
    1. 函数名严格按照 name 字段
    2. 函数的功能严格按照 description 字段
    3. 函数的参数个数严格按照 params 字段（可以有多个值）
    4. 函数的参数类型、是否必选以及返回值类型严格按照 params 和 return 字段的规定
    5. 实现的函数要严格包含函数的参数、默认值（如果有）、返回值类型以及文档字符串、行间注释
    6. 函数必须有文档字符串且格式严格按照下面的示例：
        1.分成三部分：[description]、[params]、[return]
        2.[description]：用一段纯文字详细描述函数的功能
        3.[params]：描述函数的参数，是无序列表，每一项为：- 参数名(参数类型):对参数的描述
        4.[return]：描述函数的返回值，格式为：返回值(类型):对返回值描述，只有一行
    
        例如：
        def func(a: int, b: int) -> int:
            '''
            [description]
                用于两个整数求和
            [parameters]
                - a(int):整数a
                - b(int):整数b
            [return]
                weather(str):天气信息
            '''
            return a + b
  
    7.  代码部分包裹在 
        ```python
            ...
        ```
        这样的格式中，并且不要输出任何多余的内容
"""
            ),
            MsgRole.user(
f"""
指定的 json 格式的提示是：
{self.metadata}
"""
            )
        ]
        # 生成函数代码
        llm_response = system_llm_response(
            model="gpt-4o-mini",
            messages=llm_prompt
        )
        info("函数代码已生成")
        print(llm_response)
        # 提取函数代码
        code_content = llm_response.split("```python")[1].split("```")[0].strip()
        
        # 从pkl文件加载现有函数字典，如果文件不存在则创建新字典
        try:
            with open(FUNCTION_CACHE, 'rb') as f:
                functions_dict = pickle.load(f)
        except (FileNotFoundError, EOFError):
            functions_dict = {}
        
        # 将新函数添加到字典
        functions_dict[self.metadata['name']] = code_content
        
        # 保存更新后的字典
        with open(FUNCTION_CACHE, 'wb') as f:
            pickle.dump(functions_dict, f)
        
        # 执行代码获取函数对象
        namespace = {}
        exec(code_content, namespace)
        func = namespace[self.metadata['name']]
        return func
        
    def __call__(self, *args, **kwargs):
        """ 调用函数 """
        # 首先验证当前函数是否有实际作用
        if not self.validate_actual_work(self.func):
            # 尝试从pkl文件加载函数
            try:
                with open(FUNCTION_CACHE, 'rb') as f:
                    functions_dict = pickle.load(f)
                    if self.metadata['name'] in functions_dict:
                        code_content = functions_dict[self.metadata['name']]
                        # 执行代码获取函数对象
                        namespace = {}
                        exec(code_content, namespace)
                        self.func = namespace[self.metadata['name']]
                        info("已从缓存中加载函数")
                    # 如果字典中没有这个函数且允许自动实现
                    elif self.auto_impl:
                        self.func = self.auto_implement()
                        self.generate_metadata()
                        info("函数实现时间：", self.impl_time_elapsed)
                    else:
                        raise toolserr.ToolMetaDataInitializeError("函数未实现")
            except (FileNotFoundError, EOFError):
                # 如果pkl文件不存在或为空且允许自动实现
                if self.auto_impl:
                    self.func = self.auto_implement()
                    self.generate_metadata()
                    info("函数实现时间：", self.impl_time_elapsed)
                else:
                    raise toolserr.ToolMetaDataInitializeError("函数未实现")
        
        # 计时并执行函数
        start_time = perf_counter()
        result = self.func(*args, **kwargs)
        end_time = perf_counter()
        self.time_elapsed = end_time - start_time
        info("执行时间：", self.exec_time_elapsed)
        return result
    
    
if __name__ == "__main__":
    # python -m catena_ai.catena_core.tools.function
    # def format(input: str, repeat: int = 2) -> str:
    #     """ 将输入字符串的第一个字母按照repeat次重复后输出 """
    
    # f = Function(
    #     name="a",
    #     description="ghaiwksrgagr",
    #     func=format,
    #     strict_check=True,
    #     auto_impl=True
    # )
    
    # print(f.metadata)
    # result = f("hello")
    # print(result)

    print(FUNCTION_CACHE)