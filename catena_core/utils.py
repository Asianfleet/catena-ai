import os
import cv2
import base64
import string
import random
import numpy as np

from .settings import settings
from ....paths import System

class PromptRole:
    """
    - 用于处理和包装大模型对话提示信息的工具类。功能是将不同角色的对话内容（例如系统、用户、助手等）格式化。
    - 该类包含多种方法，支持文本的预处理、角色标签的赋值以及对话上下文的处理，确保不同角色的输入能够被正确地传递给模型进行响应。
    - 此外还扩展了对图像和文本混合输入的支持，能够将用户提供的图像转换为 base64 编码，并将其与文本信息一起发送给模型。
    """

    @classmethod
    def preprocess(cls, content: str = None):
        return content.strip().strip("\t")

    @classmethod
    def mask(cls, content: str = None) -> dict:
        content = content or "你是一位优秀的助手"
        return {"role": "system", "content": cls.preprocess(content)}
    
    @classmethod
    def context(cls, content: str) -> dict:
        content = "上下文信息：\n" + content
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def system(cls, content: str) -> dict:
        return {"role": "system", "content": cls.preprocess(content)}

    @classmethod
    def assistant(cls, content: str) -> dict:
        return {"role": "assistant", "content": cls.preprocess(content)}
    
    @classmethod
    def user(cls, content: str) -> dict:
        return {"role": "user", "content": cls.preprocess(content)}
    
    @classmethod
    def user_vision(cls, text: str, image:list) -> dict:
        from src.modules.imglib.imgproclib import imcoding
        from .utils import concat_images
        if settings.prompt.image_concat_direction:
            direction = settings.prompt.image_concat_direction
            conact_savepath = System.DEBUG_DATA_PATH.val("image") + "/concated.png"
            img_concated = concat_images(image, direction, conact_savepath)
            image_base64 = imcoding.to_base64(img_concated)
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
                image_base64 = imcoding.to_base64(img)
                msg["content"].append({
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
                })
            msg["content"].append({"type": "text", "text": text})
        return msg

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    解码 base64 字符串为 OpenCV 图像格式。
    :param base64_string: base64 字符串
    :return: OpenCV 图像
    """
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def resize_image(img: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
    """
    将图像按指定宽度或高度进行等比缩放。
    :param img: 输入图像
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 调整后的图像
    """
    h, w = img.shape[:2]
    if target_width and target_height is None:
        scale = target_width / w
        new_size = (target_width, int(h * scale))
    elif target_height and target_width is None:
        scale = target_height / h
        new_size = (int(w * scale), target_height)
    else:
        return img
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def concat_images(image_list: list[str], direction: str, save_path: str):
    """
    拼接图像：上下（UD）或左右（LR）。
    :param image_list: 包含图像路径或 Base64 字符串的列表
    :param direction: 拼接方向，"UD" 为上下拼接，"LR" 为左右拼接
    :param save_path: 拼接后图像的保存路径
    """
    
    if len(image_list) < 2:
        return image_list[0]
    
    images = []

    # 加载图片
    for item in image_list:
        if os.path.isfile(item):
            img = cv2.imread(item)
        else:
            try:
                img = decode_base64_to_image(item)
            except Exception as e:
                raise ValueError(f"无法解码 base64 字符串或读取文件：{item}, 错误：{e}")
        
        if img is None:
            raise ValueError(f"无法加载图像：{item}")
        images.append(img)

    # 根据方向调整图像大小
    if direction == "LR":
        # 左右拼接，统一高度
        max_height = max(img.shape[0] for img in images)
        resized_images = [resize_image(img, target_height=max_height) for img in images]
        concatenated_image = cv2.hconcat(resized_images)
    elif direction == "UD":
        # 上下拼接，统一宽度
        max_width = max(img.shape[1] for img in images)
        resized_images = [resize_image(img, target_width=max_width) for img in images]
        concatenated_image = cv2.vconcat(resized_images)
    else:
        raise ValueError("方向参数必须为 'UD' (上下拼接) 或 'LR' (左右拼接)")

    # 保存拼接后的图像
    cv2.imwrite(save_path, concatenated_image)
    print(f"拼接后的图像已保存到：{save_path}")
    
    return concatenated_image

def random_string(length: int = 4):
    """ 生成指定长度的随机字符串 """
    # 定义包含所有大小写字母和数字的字符集
    characters = string.ascii_letters + string.digits
    # 生成随机字符串
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def info(*values):
    """ 输出运行信息 """
    if settings.debug.enable_func_level_info:
        print(*values)
 
def debug(*values):
    """ 输出调试信息 """
    if settings.debug.enable_func_level_debug:
        print(*values)

# 示例调用
if __name__ == "__main__":
    # 示例图片路径或 Base64 字符串
    image_paths = [
        "/home/legion4080/AIPJ/MYXY/background.png",  # 替换为实际图片路径
        "/home/legion4080/AIPJ/MYXY/boy.png"   # 替换为实际图片路径
    ]
    save_path = "concated.png"
    concat_images(image_paths, direction="UD", save_path=save_path)
