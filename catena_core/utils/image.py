import os
import io
import cv2
import base64
import numpy as np
from PIL import Image

from typing import (
    Union, 
    List, 
)

def to_base64(image: Union[str, np.ndarray, Image.Image]) -> str:
    # 如果输入是base64字符串，直接返回
    if isinstance(image, str):
        # 判断输入字符串是否已是base64编码
        if not os.path.isfile(image):
            return image
        with open(image, "rb") as img_file:
            # 读取图片并进行base64编码
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    # 如果输入是NumPy数组
    elif isinstance(image, np.ndarray):
        # 确保数组是图像数据类型
        pil_img = Image.fromarray(image)
        return to_base64(pil_img)
    
    # 如果输入是PIL的Image对象
    elif isinstance(image, Image.Image):
        # 将PIL图像保存到内存中的字节流
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        # 获取图片的base64编码
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 如果输入类型不支持，抛出异常
    else:
        raise TypeError("Unsupported image type")

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

def concat_images(image_list: List[str], direction: str, save_path: str):
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

