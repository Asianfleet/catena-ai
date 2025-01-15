import os
import re
from catenaconf import Catenaconf
from pydantic import BaseModel

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
src_folder = os.path.dirname(current_script_path)
PROJ_BASEPATH = os.path.dirname(src_folder)
###################################################
#             Load Configs from YAML              #
###################################################
def loadconf(config: str, search_path: str = os.path.join(PROJ_BASEPATH, "configs")):
    """ 从配置文件加载配置 """
    config_path = os.path.join(search_path, config.replace(".", "/") + ".yaml")
    return Catenaconf.load(config_path)

def loadstructconf(config: BaseModel):
    return Catenaconf.structured(config)

###################################################
#                Basic Path Class                #
###################################################

class Path:
    """ 
    - 路径拼接：通过构造函数 __init__,可以根据提供的 base_path 和 value 动态生成路径。
    - 参数替换：通过 val 方法，可以将路径字符串中的占位符（例如 $name$）替换为实际的参数值。
    """
    def __init__(self, value: str, base_path: str = None):
        self.value = os.path.join(str(base_path), value) if base_path else value

    def val(self, *arg):
        
        params = list(arg)
        if not params:
            assert self.value.find('$') == -1, "没有传入参数"
            return self.value
        pattern = r'\$[^\$]*\$'
     
        return re.sub(pattern, lambda match: str(params.pop(0)), self.value)
    
    def __call__(self):

        return self.value
    
###################################################
#             Comic Consts Configs                #             
###################################################
class Comiconsts:

    BASEPATH = os.path.join(PROJ_BASEPATH, "MYXY/scripts/output/")
    DATABASEPATH = os.path.join(PROJ_BASEPATH, "MYXY/scripts/data/comics/ComicBook.db")
    PGPATH = os.path.join(PROJ_BASEPATH, "MYXY/scripts/output/pages/")
    BOOKPATH = os.path.join(PROJ_BASEPATH, "MYXY/data/comics/books")
    SCRP_PATH = Path("$name$/scripts/scripts.yaml", BOOKPATH)
    OUTL_PATH = Path("$name$/scripts/ontline.yaml", BOOKPATH)
    CMC_STAT = Path("$name$/status.yaml", BOOKPATH)

###################################################
#             Other Consts Configs                #             
###################################################
class Config:
    def __init__(self, config: dict):
        self.__dict__ = config

class System:
    DEBUG_DATA_PATH = Path("MYXY/data/debug/$type$", PROJ_BASEPATH)

class ConfPaths:
    CONF_PATH = os.path.join(PROJ_BASEPATH, "MYXY/configs")
    CONFF_PATH = Path("$filefolder$/$configfile$.yaml", CONF_PATH)
    LOG_PATH = os.path.join(PROJ_BASEPATH, "MYXY/configs/")
    DATA_SEND = Path("data/transfer/send/$task_id$", PROJ_BASEPATH)
    DATA_RECV = Path("data/transfer/recv/$task_id$", PROJ_BASEPATH)

class ServConf:
    REMOTE = Config({
        "INFER": Config({"IP": "aimh.e6.luyouxia.net", "PORT": "46335"}), 
        "SSH": Config({"IP": "cn-hk-bgp-4.ofalias.net", "PORT": "21173"})
    })
    USERNAME = "admin1"
    PASSWORD = "sduwhai"

    INPUT_PATH = Path("/data/AIMH/AgentCtrl/Input/$name$")
    LOG_PATH = Path("/data/AIMH/AgentCtrl/Logs/$name$/$task_id$.log")
    TD_PATH = Path("/data/AIMH/AIMHWorkspace/data/3Dmodel/$task_id$")
    PY_ITP = Path("/data/envs/$name$/bin")
    
class BlenderConf:
    PYITP = '/data/AIMH/TD/blender/blender --background --python'
    CONSOLE = '/data/AIMH/TD/blender/blender --background --python-console'
    SCRIPTS = '/data/AIMH/TD/blender/4.1/python/scripts/'
 
class llmconf:
    DB_PATH = os.path.join(PROJ_BASEPATH, "src/modules/catena/retriever/db")
    PROMPT_PATH = os.path.join(PROJ_BASEPATH, "src/modules/catena/data/prompt")
    PATTERN_PATH = os.path.join(PROJ_BASEPATH, "src/modules/catena/data/pattern")

    DB_PATH_ = Path("src/modules/catena/retriever/db/$name$", PROJ_BASEPATH)

class modelconf:
    PROMPT_EXPANSION = Path("src/models/GPT-Prompt-Expansion-Fooocus-v2", PROJ_BASEPATH).val()

class AIDW:
    DS_SYN = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    DS_TASK = Path("https://dashscope.aliyuncs.com/api/v1/tasks/$task_id$")
    WEBUI_BASE_URL = {
        "3080": "http://192.168.31.123:7860",
        "A100": "http://192.168.31.123:7860"
    }

class ComicScene:
    SCENE_PATH = Path("$book_name$/scenes/scene.blend", Comiconsts.BOOKPATH)

class AgentCfg:
    SAVE_PATH = Path("src/modules/catena/data/saves/$name$.yaml", PROJ_BASEPATH)

class GradioCfg:
    BASE_PATH = os.path.join(PROJ_BASEPATH, "data/frontend/comics")
    ASSETS_PATH = Path("data/frontend/comics/$name$/assets", PROJ_BASEPATH)
    PAGE_PATH = Path("data/frontend/comics/$name$/pages", PROJ_BASEPATH)
    SCRIPT_PATH = Path("data/frontend/comics/$name$/script.json", PROJ_BASEPATH)
    SETTING_PATH = Path("data/frontend/comics/$name$/setting.md", PROJ_BASEPATH)
    STORY_PATH = Path("data/frontend/comics/$name$/story.txt", PROJ_BASEPATH)
    PLAN_PATH = Path("data/frontend/comics/$name$/scene_plan.csv", PROJ_BASEPATH)
    PREVIEW_PATH = Path("data/frontend/comics/$name$/scene_preview.csv", PROJ_BASEPATH)
    MAPPING_PATH = Path("data/frontend/comics/$name$/assets/mapping.json", PROJ_BASEPATH)
    MAPPING_PIC_PATH = Path("data/frontend/comics/$name$/pages/mapping.json", PROJ_BASEPATH)
    FINAL_SCRIPT_MAPPING_PATH = Path("data/frontend/comics/$name$/final_script_mapping.json", PROJ_BASEPATH)
    RENDERED_PATH = Path("data/frontend/comics/$name$/rendered.json", PROJ_BASEPATH)

if __name__ == '__main__':
    print(PROJ_BASEPATH)
    print(ServConf.REMOTE.INFER.IP)