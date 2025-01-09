import re
import copy
from typing import Any

class KvConfig(dict):
    def __init__(self, *args, **kwargs):
        """ Initialize the KvConfig class, and the internal nested dictionary will also be converted to the KvConfig type """
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = KvConfig(value)
            elif isinstance(value, list):
                self[key] = [KvConfig(item) if isinstance(item, dict) else item for item in value]

    # TODO: the KvConfig class may have special attributes with underlines, 
    # which can't accessd by super().__getattr__(key)
    def __getattr__(self, key):
        """ Get the value of the key """
        
        # The following two lines of code seems to be useless
        """ if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key) """

        try:
            value = self[key]
            # Return directly (the init function ensures that it is already of KvConfig type)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """ Set the value of the key """
        
        # make sure the special attributes are not overwritten by the key-value pair
        if key.startswith('__') and key.endswith('__'):
            super().__setattr__(key, value)
        else: 
            # Ensure that after adding new attributes, they will also be converted to KvConfig type
            if isinstance(value, dict):
                value = KvConfig(value)
            elif isinstance(value, list):
                value = [KvConfig(item) if isinstance(item, dict) else item for item in value]
        
            self[key] = value

    def __delattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        """ Make a deep copy of an instance of the KvConfig class """
        # Use the default dict copying method to avoid infinite recursion.
        return KvConfig(copy.deepcopy(dict(self), memo))

    @property
    def deepcopy(self):
        """ Make a deep copy of an instance of the KvConfig class """
        return copy.deepcopy(self)  
    
    @property
    def __ref__(self):
        return self.__getallref__()
    
    def __getallref__(self):
        return re.findall(r'@\{(.*?)\}', self.__str__())
    
    @property
    def __container__(self) -> dict:
        """ Copy the KvConfig instance, convert it to a normal dict and output """
        return self.__to_container__()

    def __to_container__(self) -> dict:
        """ Copy the KvConfig instance, convert it to a normal dict and output """
        self_copy = self.deepcopy
        for key, value in self_copy.items():
            if isinstance(value, KvConfig):
                self_copy[key] = value.__to_container__()
            elif isinstance(value, dict):
                self_copy[key] = KvConfig(value).__to_container__()
        return dict(self_copy)

class Catenaconf:
    @staticmethod
    def create(config: dict) -> KvConfig:
        """ Create a KvConfig instance """
        return KvConfig(config)

    @staticmethod
    def update(cfg: KvConfig, key: str, value: Any = None, *, merge: bool = True) -> None:
        keys = key.split('.')
        current = cfg
        for k in keys[:-1]:
            if k not in current:
                current[k] = KvConfig({})
            current = current[k]
        last_key = keys[-1]

        if merge:
            if isinstance(current.get(last_key, KvConfig({})), KvConfig):
                if isinstance(value, dict) or isinstance(value, KvConfig):
                    for k, v in value.items():
                        current[last_key][k] = v
                    current[last_key] = KvConfig(current[last_key])
                else:
                    current[last_key] = value
            else:
                    current[last_key] = value
        else:
            if isinstance(value, dict):
                current[last_key] = KvConfig(value)
            else:
                current[last_key] = value

    @staticmethod
    def merge(*configs) -> KvConfig:
        
        def merge_into(target: KvConfig, source: KvConfig) -> None:
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_into(target[key], value)
                else:
                    target[key] = value
                    
        merged_config = KvConfig({})
        for config in configs:
            merge_into(merged_config, KvConfig(config))
        return KvConfig(merged_config)

    @staticmethod
    def resolve(cfg: KvConfig) -> None:
        capture_pattern = r'@\{(.*?)\}'
        def de_ref(captured):
            ref:str = captured.group(1)
            target = cfg
            for part in ref.split("."):
                target = target[part]
            return str(target)

        def sub_resolve(input: KvConfig):
            for key, value in input.items():
                if isinstance(value, KvConfig):
                    sub_resolve(value)
                elif isinstance(value, str):
                    if re.search(capture_pattern, value):
                        content = re.sub(capture_pattern, de_ref, value)
                        input[key] = content

        sub_resolve(cfg)

    @staticmethod
    def to_container(cfg: KvConfig, resolve = True) -> dict:
        """ convert KvConfig instance to a normal dict and output. """
        if resolve:
            cfg_copy = cfg.deepcopy
            Catenaconf.resolve(cfg_copy)
            return cfg_copy.__to_container__()
        else:
            return cfg.__to_container__()

# 测试代码
test = {
    "config": {
        "database": {
            "host": "localhost",
            "port": 5432
        },
        "connection": "Host: @{config.database.host}, Port: @{config.database.port}"
    },
    "app": {
        "version": "1.0.0",
        "info": "App Version: @{app.version}, Connection: @{config.connection}"
    }
}

if __name__ == "__main__":
    #print(test)
    test = {"a":"a"}
    dt = KvConfig(test)
    """ 
    dt = Catenaconf.create(test)
    Catenaconf.resolve(dt)
    print(dt)

    dt.config.database.host = "123"
    print(dt)

    Catenaconf.update(dt, "config.database", {"123": "123"})
    print(dt)

    ds = Catenaconf.merge(dt, {"new_key": "new_value"})
    print(ds)"""
    
    #Catenaconf.update(dt, "config.database.host", "4567")
    #print(dt) 
    
    """ dt.config.database.host = "4567"
    print(dt) """
    print(dt.__class__)
    print(dt.a)