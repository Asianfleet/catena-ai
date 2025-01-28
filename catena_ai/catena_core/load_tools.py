import os
from typing import Union, List
from catenaconf import Catenaconf, KvConfig
from ..catena_core.paths import llmconf

from catenaconf import Catenaconf

def load_prompt(label: str, resolve: bool = False):
    if len(label.split(".")) == 2:
        file = label.split(".")[0]
        label = label.split(".")[1]
        config_path = os.path.join(llmconf.PROMPT_PATH, file + ".yaml")
        prompt = Catenaconf.load(config_path)[label]
        if resolve: Catenaconf.resolve(prompt)
        return prompt
    else:
        return "invalid label"

def load_pattern(label: str, resolve: bool = False) -> Union[KvConfig]:
    file = label.split(".")[0]
    label = label.split(".")[1]
    config_path = os.path.join(llmconf.PATTERN_PATH, file + ".yaml")
    pattern = Catenaconf.load(config_path)[label]
    if resolve: Catenaconf.resolve(pattern)
    return pattern
    
#TODO： 逐步移除该函数
def interpolate(config: Union[KvConfig, dict, str]):
    if isinstance(config, KvConfig):
        Catenaconf.resolve(config)

#TODO： 逐步移除该函数
def unwarp(config: Union[KvConfig, dict]) -> Union[dict, List]:
    if isinstance(config, KvConfig):
        return Catenaconf.to_container(config)
    else:
        return config
    
if __name__ == "__main__":
    prompt = load_prompt("prompt.prompt")
    print(prompt)