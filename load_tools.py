import os
from typing import Union
from omegaconf import OmegaConf, ListConfig, DictConfig
from src.paths import llmconf

def load_prompt(label: str, resolve: bool = False):
    if len(label.split(".")) == 2:
        file = label.split(".")[0]
        label = label.split(".")[1]
        config_path = os.path.join(llmconf.PROMPT_PATH, file + ".yaml")
        prompt = OmegaConf.load(config_path)[label]
        if resolve: OmegaConf.resolve(prompt)
        return prompt
    else:
        return "invalid label"

def load_pattern(label: str, resolve: bool = False) -> Union[ListConfig, DictConfig]:
    file = label.split(".")[0]
    label = label.split(".")[1]
    config_path = os.path.join(llmconf.PATTERN_PATH, file + ".yaml")
    pattern = OmegaConf.load(config_path)[label]
    if resolve: OmegaConf.resolve(pattern)
    return pattern
    
#TODO： 逐步移除该函数
def interpolate(config: Union[ListConfig, DictConfig, dict, str]):
    if isinstance(config, DictConfig) or isinstance(config, ListConfig):
        OmegaConf.resolve(config)

#TODO： 逐步移除该函数
def unwarp(config: Union[ListConfig, DictConfig, dict]) -> Union[dict, list]:
    if isinstance(config, DictConfig) or isinstance(config, ListConfig):
        return OmegaConf.to_container(config)
    else:
        return config