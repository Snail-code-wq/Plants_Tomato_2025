import json
import os
import cv2
import base64

def encode_image_to_base64(image_path):
    """将图像编码为 base64 格式"""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def update_image_path(json_folder):
    for file in os.listdir(json_folder):
        if file.endswith('.json'):
            json_path = os.path.join(json_folder, file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取文件名并更新 imagePath
            new_image_name = os.path.splitext(file)[0] + '.jpg'  # 假设对应的图像是 PNG 格式
            data['imagePath'] = new_image_name  # 更新为新文件名

            # 更新 imageData
            image_path = os.path.join(json_folder, new_image_name)  # 新的图像路径
            if os.path.exists(image_path):
                data['imageData'] = encode_image_to_base64(image_path)  # 更新为图像的 base64 编码
            else:
                print(f"未找到对应的图像文件: {image_path}")

            # 写入回 JSON 文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

# 使用方法
json_folder = r'D:\deep\data\scar\var\scale'  # JSON 文件所在文件夹路径
update_image_path(json_folder)
