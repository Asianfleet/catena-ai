from typing import (
    Union, 
    List, 
    Dict
)

from catena.catena_core.paths import System
from .image import to_base64
from ...settings import settings

class MessageRole:
    """
    - 用于处理和包装大模型对话提示信息的工具类。功能是将不同角色的对话内容（例如系统、用户、助手等）格式化。
    - 该类包含多种方法，支持文本的预处理、角色标签的赋值以及对话上下文的处理，确保不同角色的输入能够被正确地传递给模型进行响应。
    - 此外还扩展了对图像和文本混合输入的支持，能够将用户提供的图像转换为 base64 编码，并将其与文本信息一起发送给模型。
    """

    @classmethod
    def preprocess(cls, content: str = None):
        return content.strip().strip("\t")

    @classmethod
    def mask(cls, content: str = None) -> Dict:
        content = content or "你是一位优秀的助手"
        return {"role": "system", "content": cls.preprocess(content)}
    
    @classmethod
    def context(cls, content: str) -> Dict:
        content = "上下文信息：\n" + content
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def system(cls, content: str) -> Dict:
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def assistant(cls, content: str) -> Dict:
        return {"role": "assistant", "content": cls.preprocess(content)}
    
    @classmethod
    def user(cls, content: str) -> Dict:
        return {"role": "user", "content": cls.preprocess(content)}
    
    @classmethod
    def user_vision(cls, text: str, image:List) -> Dict:
        from .utils import concat_images
        if settings.prompt.image_concat_direction:
            direction = settings.prompt.image_concat_direction
            conact_savepath = System.DEBUG_DATA_PATH.val("image") + "/concated.png"
            img_concated = concat_images(image, direction, conact_savepath)
            image_base64 = to_base64(img_concated)
            msg = {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                },
                {"type": "text", "text": text}
            ]}
        else:
            msg = {"role": "user", "content": []}
            for img in image:
                image_base64 = to_base64(img)
                msg["content"].append({
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                })
            msg["content"].append({"type": "text", "text": text})
        return msg

# 示例调用
if __name__ == "__main__":
    # 示例图片路径或 Base64 字符串
    image_paths = [
        "/home/legion4080/AIPJ/MYXY/background.png",  # 替换为实际图片路径
        "/home/legion4080/AIPJ/MYXY/boy.png"   # 替换为实际图片路径
    ]
    
