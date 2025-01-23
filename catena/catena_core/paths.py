import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
src_folder = os.path.dirname(current_script_path)
LIB_BASEPATH = os.path.dirname(src_folder)

class Template:
    PROMPT_PATH = os.path.join(LIB_BASEPATH, ".data", ".templates")
    
