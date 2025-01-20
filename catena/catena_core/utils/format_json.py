import json
from typing import (
    Dict, 
    List, 
    Union
)

def delete_json_format(input: Union[str, Dict]) -> str:
    """ 将json格式的字符串转化为无格式的字符串 """

    # 确保将input转化为字典
    if isinstance(input, Dict): 
        obj = input 
    elif isinstance(input, str):        
        try:
            obj = json.loads(input)
        except json.JSONDecodeError as e:
            print(f"[delete_json_format] JSONDecodeError: {e}")

    # 递归处理字典，将其转化为无格式的字符串
    def remove_quotes(obj: Union[str, Dict]):

        json_str_no_quotes = ""
        if isinstance(obj, str):
            return obj
        for key, value in obj.items():
            if isinstance(value, Dict):
                json_str_no_quotes += f"{key}: {remove_quotes(value)}"
            elif isinstance(value, List):
                json_str_no_quotes += f"{key}: "
                for item in value:
                    json_str_no_quotes += f"{remove_quotes(item)}"
            else:
                json_str_no_quotes += f"{key}: {value}"
                
        return json_str_no_quotes

    return remove_quotes(obj)