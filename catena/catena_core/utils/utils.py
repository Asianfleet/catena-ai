import re
import base64
from typing import (
    Union, 
    List, 
    Dict
)

class TemplateCreater:
    
    def __init__(self, name):
        self.name = name
        self.message = []
        self.meta = {"param": {}}
        self.temlpate = {}
        
    def new_message(self, role: str, content: str, **kwargs):
        
        pattern = r"@\{(.*)\}"
        def format_param(match: re.Match):
            matched = match.group(1)
            return f"@{{{self.name}.meta.param.{matched}}}"
        
        self.message.append({
            "role": role,
            "content": re.sub(pattern, format_param, content),
            **kwargs
        })
        
        match = re.findall(pattern, content)
        for param in match:
            self.meta["param"][param] = None
        
    def add_meta(self, meta: Dict):
        self.meta.update(meta)
        
    def create_template(self):
        self.temlpate = {
            self.name: {
                "message": self.message,
                "meta": self.meta
            }
        }
        
        return self.temlpate

def is_url_or_base64(s):
    # 判断是否是URL
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if re.match(url_pattern, s):
        return "URL"
    
    # 判断是否是Base64
    try:
        # 尝试解码Base64
        base64.b64decode(s, validate=True)
        return "Base64"
    except:
        pass
    
    return "Neither URL nor Base64"


# 示例调用
if __name__ == "__main__":
    # 示例图片路径或 Base64 字符串
    # python -m catena.catena_core.utils.utils
    image_paths = [
        "/home/legion4080/AIPJ/MYXY/background.png",  # 替换为实际图片路径
        "/home/legion4080/AIPJ/MYXY/boy.png"   # 替换为实际图片路径
    ]
    
    tpl = TemplateCreater("test")
    tpl.new_message("user", "你好@{name}")
    tpl.add_meta({"dfF":"Fawfs"})
    print(tpl.create_template())