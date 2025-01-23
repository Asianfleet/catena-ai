import os
from enum import Enum
from catenaconf import Catenaconf

class BtnTemplate(Enum):
    
    multimodal = "multimodal.yaml"

def load_prompt(label: Enum, resolve: bool = False):
    if not isinstance(label, Enum):
        return "invalid label"
    if len(label.value.split(".")) == 2:
        file = label.split(".")[0]
        label = label.split(".")[1]
        config_path = os.path.join(llmconf.PROMPT_PATH, file + ".yaml")
        prompt = Catenaconf.load(config_path)[label]
        if resolve: Catenaconf.resolve(prompt)
        return prompt
    else:
        return "invalid label"

