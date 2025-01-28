import string
import random
from typing import List

def delete_list_format(input: List, seperator: str = "\n") -> str:
    """ 将列表格式的字符串转化为无格式的字符串 """
    
    return seperator.join([str(item) for item in input])

def random_string(length: int = 4):
    """ 生成指定长度的随机字符串 """
    # 定义包含所有大小写字母和数字的字符集
    characters = string.ascii_letters + string.digits
    # 生成随机字符串
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string