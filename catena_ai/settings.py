from __future__ import annotations
import os
import copy
import threading
from typing import Union
from copy import deepcopy
from dataclasses import dataclass
from contextlib import contextmanager
from catenaconf import Catenaconf, KvConfig
    
# 默认配置
DEFAULT_CONFIG = Catenaconf.create({           
    "concurrecy_limit": 10,          # 并发限制
})

PROMPT_CONFIG = Catenaconf.create({
    "image_concat_direction": None,  # 图片拼接方向（UD-上下；LR-左右；为空则不进行拼接）
})

LM_CONFIG = Catenaconf.create({
    "model_name": "",
    "enable_lm_refinement": False,   # 是否启用提示词优化
})     

LLM_CONFIG = Catenaconf.create({
    #           平台设置           
    ""                                          # "https://api.agicto.cn/v1"
    "thirdparty_url": os.getenv("TP_URL"),      # 第三方平台 URL
    "thirdparty_api_key": os.getenv("TP_KEY"),  # 第三方平台 API 密钥
    "enable_token_count": False,                # 是否启用 token 计算
    #           模型参数           
    "max_tokens": 2048,                         # 最大 token 数
    "temperature": 0.3,                         # 温度
    "top_p": 0.6,                               # Top-p 采样
    
})

LOG_CONFIG = Catenaconf.create({
    "enable_chain_visualize": True,     # 可视化管道内部过程
    "enable_func_level_debug": False,   # 输出函数级别的调试信息
    "enable_func_level_info": True,     # 输出函数级别的信息
    "enable_func_level_warning": True,  # 输出函数级别的警告
    "enable_func_level_error": True     # 输出函数级别的错误
})

VISUALIZE_CONFIG = Catenaconf.create({
    "visualize_type": "tree",
    "metrics": False,
    "message_metrics": False,
    "model_resp_metrics": False
})

STYLE_CONFIG = Catenaconf.create({
    "chain_start": "[bold green]",
    "chain_end": "[bold gray]",
    "chain_spacer": "▬",
    "spacer_repeat": 50,
    "completion_mark": "-"
})

AGENT_CONFIG = Catenaconf.create({
    "agent_type": "patterned",
    "pattern": "default",
    "enable_agent_visualize": True,
    "enable_agent_debug": False,
    "enable_agent_info": True
})

MINIMAL_LLM_CONFIG = Catenaconf.create({
    "max_tokens": 1024,
    "temperature": 0.3,
    
})

class BaseSettings:
    """ configuration settings."""

    _instance = None
    _DEFAULT_CONFIG = DEFAULT_CONFIG
    
    def __new__(cls, CONFIG = None):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.lock = threading.Lock()
            cls._instance.main_tid = threading.get_ident()
            cls._instance.main_stack = []
            cls._instance.stack_by_thread = {}
            cls._instance.stack_by_thread[threading.get_ident()] = cls._instance.main_stack

            # 使用 Catenaconf 创建配置
            DEFAULT_CONFIG = CONFIG or cls._DEFAULT_CONFIG
            cls._instance.__append(deepcopy(DEFAULT_CONFIG))

        return cls._instance
    
    def __init__(self, CONFIG = None):
        if CONFIG:
            self._DEFAULT_CONFIG = CONFIG

    @property
    def config(self) -> KvConfig:
        thread_id = threading.get_ident()
        if thread_id not in self.stack_by_thread:
            main_stack_copy = deepcopy(self.main_stack[-1])
            self.stack_by_thread[thread_id] = [Catenaconf.create(main_stack_copy)]
        return self.stack_by_thread[thread_id][-1]

    def __getattr__(self, name):
        if name in self.config:
            return Catenaconf.select(self.config, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __append(self, config):
        thread_id = threading.get_ident()
        if thread_id not in self.stack_by_thread:
            self.stack_by_thread[thread_id] = [Catenaconf.create(self.main_stack[-1])]
        self.stack_by_thread[thread_id].append(config)

    def __pop(self):
        thread_id = threading.get_ident()
        if thread_id in self.stack_by_thread:
            self.stack_by_thread[thread_id].pop()

    def configure(self, inherit_config: bool = True, **kwargs):
        """Set configuration settings.

        Args:
            inherit_config (bool, optional): Set configurations for the given, and use existing configurations for the rest. Defaults to True.
        """
        if inherit_config:
            config = Catenaconf.merge(self.config, Catenaconf.create(kwargs))
        else:
            config = Catenaconf.create(kwargs)

        self.__append(config)

    @contextmanager
    def context(self, inherit_config=True, **kwargs):
        self.configure(inherit_config=inherit_config, **kwargs)

        try:
            yield
        finally:
            self.__pop()
        
    def __repr__(self) -> str:
        return repr(self.config)

class PromptSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = PROMPT_CONFIG
 
class LMSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = LM_CONFIG
 
class LLMSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = LLM_CONFIG
 
class DebugSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = LOG_CONFIG
 
class VisualizeSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = VISUALIZE_CONFIG

class StyleSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = STYLE_CONFIG

class AgentSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = AGENT_CONFIG

class Minimal_LLMSettings(BaseSettings):
    _instance = None
    _DEFAULT_CONFIG = MINIMAL_LLM_CONFIG

@dataclass
class Settings:
    """ 存储所有配置的类 """
    base: BaseSettings = BaseSettings()
    lm: BaseSettings = LMSettings()
    llm: BaseSettings = LLMSettings()
    debug: BaseSettings = DebugSettings()
    visualize: BaseSettings = VisualizeSettings()
    prompt: BaseSettings = PromptSettings()
    style: BaseSettings = StyleSettings()
    agent: BaseSettings = AgentSettings()
    minimal_llm: BaseSettings = Minimal_LLMSettings()

settings = Settings()

#TODO: 将大模型管道组件的配置参数 config 类型由泛型类 Runtimeconfig 改为 RTConfig
# 1、增强类型检查
# 2、提高数据传输的灵活性

class RTConfig:
    """ 运行时配置，用于在组件之间运行时动态传递配置 """
    def __init__(self, config: dict={}):
        self._config = Catenaconf.create(config)
    
    def __call__(self, keys: str=None, deep_copy: bool=True) -> KvConfig:
        """ 获取配置，提供键值访问以及深拷贝功能 """
        if keys:
            if not deep_copy:
                config_return = Catenaconf.select(self._config, keys)
            else:
                config_return = copy.deepcopy(Catenaconf.select(self._config, keys))
        else:
            config_return = copy.deepcopy(self._config) if deep_copy else self._config
        return config_return
    
    @property
    def data(self):
        """ 获取全部配置，进行深拷贝 """
        return self.__call__()
    
    @property
    def unwrap(self):
        """ 获取配置的字典形式 """
        return Catenaconf.to_container(self._config)

    def _update(self, config: Union[dict, "RTConfig", KvConfig], spec: str=None):
        if spec:
            config = config() if isinstance(config, RTConfig) else config
            Catenaconf.update(self._config, spec, config)

    def interpolate(self):
        """ 解析配置中的引用，不改变原配置  """
        config_deepcopy = copy.deepcopy(self._config)
        Catenaconf.resolve(config_deepcopy)
        return config_deepcopy
        
    def _interpolate(self, deep_copy: bool=True):
        """ 解析配置中的引用，会改变原配置 """

        Catenaconf.resolve(self._config)
        if deep_copy:
            config_return = copy.deepcopy(self._config)
        else:
            config_return = self._config
   
        return config_return
    
    def merge(self, config: Union[dict, "RTConfig", KvConfig], spec: str=None):
        """ 合并配置，不改变原配置 """
       
        full_copy = self.__call__()     # 深拷贝整个配置
        part_copy = self.__call__(spec) # 深拷贝部分配置，有可能为整个配置
  
        if isinstance(config, dict):
            part_copy = Catenaconf.merge(part_copy, Catenaconf.create(config))
        elif isinstance(config, RTConfig):
            part_copy = Catenaconf.merge(part_copy, config())
        if spec:
            Catenaconf.update(full_copy, spec, part_copy)
        else:
            full_copy = part_copy
  
        return full_copy
    
    def _merge(self, config: Union[dict, "RTConfig", KvConfig], spec: str=None):
        """ 合并配置，会改变原配置 """
        part_copy = self.__call__(spec)    # 深拷贝部分配置，有可能为整个配置
        # 合并部分配置
        if isinstance(config, dict):
            part_copy = Catenaconf.merge(part_copy, Catenaconf.create(config))
        else:
            part_copy = Catenaconf.merge(part_copy, config())
        if spec:    # 直接更新原配置
            Catenaconf.update(self._config, spec, part_copy)
        else:
            self._config = part_copy
        
        return self.data    # 返回整个配置的深拷贝
    
    def __bool__(self):
        """ 判断配置是否为空 """
        boo = self.unwrap != {}
        return boo

def info(*values):
    """ 输出运行信息 """
    if settings.debug.enable_func_level_info:
        print(*values)
 
def debug(*values):
    """ 输出调试信息 """
    if settings.debug.enable_func_level_debug:
        print(*values)

if __name__ == "__main__":
    """ class DSPySettings:
        #DSP configuration settings.

        _instance = None

        def __new__(cls):
            
            #Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
            

            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.lock = threading.Lock()
                cls._instance.main_tid = threading.get_ident()
                cls._instance.main_stack = []
                cls._instance.stack_by_thread = {}
                cls._instance.stack_by_thread[threading.get_ident()] = cls._instance.main_stack

                #  TODO: remove first-class support for re-ranker and potentially combine with RM to form a pipeline of sorts
                #  eg: RetrieveThenRerankPipeline(RetrievalModel, Reranker)
                #  downstream operations like dsp.retrieve would use configs from the defined pipeline.

                # make a deepcopy of the default config to avoid modifying the default config
                cls._instance.__append(deepcopy(DEFAULT_CONFIG))

            return cls._instance

        @property
        def config(self):
            thread_id = threading.get_ident()
            if thread_id not in self.stack_by_thread:
                self.stack_by_thread[thread_id] = [self.main_stack[-1].copy()]
            return self.stack_by_thread[thread_id][-1]

        def __getattr__(self, name):
            if hasattr(self.config, name):
                return getattr(self.config, name)

            if name in self.config:
                return self.config[name]

            super().__getattr__(name)

        def __append(self, config):
            thread_id = threading.get_ident()
            if thread_id not in self.stack_by_thread:
                self.stack_by_thread[thread_id] = [self.main_stack[-1].copy()]
            self.stack_by_thread[thread_id].append(config)

        def __pop(self):
            thread_id = threading.get_ident()
            if thread_id in self.stack_by_thread:
                self.stack_by_thread[thread_id].pop()

        def configure(self, inherit_config: bool = True, **kwargs):
            
            if inherit_config:
                config = {**self.config, **kwargs}
            else:
                config = {**kwargs}

            self.__append(config)

        @contextmanager
        def context(self, inherit_config=True, **kwargs):
            self.configure(inherit_config=inherit_config, **kwargs)

            try:
                yield
            finally:
                self.__pop()

        def __repr__(self) -> str:
            return repr(self.config) """


    settings.base.configure(tt=1)
    print(settings.base.config)
    print(settings.debug.enable_func_level_info)
    cfg = RTConfig({"a": 1, "b": 2})
    print(cfg.unwrap)
    settings.debug.configure(enable_func_level_info=True)
    info("test:", "test")
    
    """ cfg=RTConfig({"a": 1, "b": 2, "c": {"d": {"f": 5}, "e": 4}})
    cfg1 = {"g": 5, "h": 6}
    mg = cfg._merge(cfg1)
    mg.a = 2
    
    print(mg)
    print(cfg()) """
    
