import os
import cv2
import json
import base64
import numpy as np
from pathlib import Path

# 配置
folder = r"D:\deep\data\scar\train\rotation_scale"  # 图片和JSON文件所在文件夹
output_folder = r"D:\deep\data\scar\train\brighten"  # 增强后数据保存位置
os.makedirs(output_folder, exist_ok=True)

# 光照增强与减弱函数
def adjust_brightness(image, factor):
    """
    调整图片亮度。
    :param image: 输入图片 (numpy array)
    :param factor: 亮度调整因子 (>1 增强，<1 减弱)
    :return: 调整后的图片
    """
    image = image.astype(np.float32)
    image = image * factor
    image = np.clip(image, 0, 255)  # 防止像素溢出
    return image.astype(np.uint8)

# 将图像转换为 Base64 格式

def encode_image_to_base64(image_path):
    """
    将图像文件转换为 Base64 编码。
    :param image_path: 图像文件路径
    :return: Base64 字符串
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 处理每个文件
files = os.listdir(folder)
for file in files:
    if file.endswith(('.jpg', '.png', '.jpeg')):  # 检测图片文件
        basename = os.path.splitext(file)[0]
        image_path = os.path.join(folder, file)
        json_path = os.path.join(folder, basename + ".json")

        # 检查是否有对应的 JSON 文件
        if not os.path.exists(json_path):
            print(f"未找到对应 JSON 文件: {json_path}")
            continue

        # 加载图片
        image = cv2.imread(image_path)

        # 光照增强和减弱
        brighter_image = adjust_brightness(image, 1.5)  # 增强亮度
        darker_image = adjust_brightness(image, 0.7)  # 减弱亮度

        # 保存增强后的图片
        brighter_image_path = os.path.join(output_folder, basename + "_brighter.jpg")
        darker_image_path = os.path.join(output_folder, basename + "_darker.jpg")
        cv2.imwrite(brighter_image_path, brighter_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(darker_image_path, darker_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # 加载 JSON 文件
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # 更新 JSON 文件内容
        for img_type, img_path in [("brighter", brighter_image_path), ("darker", darker_image_path)]:
            new_json_data = json_data.copy()

            # 更新 imagePath 字段
            new_json_data["imagePath"] = os.path.basename(img_path)

            # 更新 imageData 字段
            new_json_data["imageData"] = encode_image_to_base64(img_path)

            # 保存更新后的 JSON 文件
            new_json_path = os.path.join(output_folder, basename + f"_{img_type}.json")
            with open(new_json_path, 'w') as f:
                json.dump(new_json_data, f, indent=4)

        print(f"已处理: {file} -> 生成亮度增强和减弱的图片及同步更新的 JSON 文件。")
