import cv2
import json
import numpy as np
import os
import base64

def rotate_image(image, angle, bg_color=(0, 0, 0)):
    """旋转图像并调整尺寸，背景颜色可指定"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新尺寸
    if angle % 180 == 90:
        rotated_image = cv2.warpAffine(image, matrix, (h, w), borderValue=bg_color)
    else:
        rotated_image = cv2.warpAffine(image, matrix, (w, h), borderValue=bg_color)

    return rotated_image

def rotate_annotations(annotations, angle, image_shape):
    """旋转标注数据并调整坐标"""
    h, w = image_shape[:2]
    new_annotations = []

    for ann in annotations:
        points = np.array(ann['points'])
        rotated_points = []

        for (x, y) in points:
            if angle == 90:
                new_x = y
                new_y = w - x
            elif angle == 180:
                new_x = w - x
                new_y = h - y
            elif angle == 270:
                new_x = h - y
                new_y = x
            elif angle in (30, 45):  # 针对30和45度的特殊处理
                angle_rad = np.radians(angle)
                new_x = int((x - w / 2) * np.cos(angle_rad) - (y - h / 2) * np.sin(angle_rad) + w / 2)
                new_y = int((x - w / 2) * np.sin(angle_rad) + (y - h / 2) * np.cos(angle_rad) + h / 2)
            else:
                new_x, new_y = x, y

            rotated_points.append([new_x, new_y])

        new_annotations.append({
            'label': ann['label'],
            'points': rotated_points,
            'group_id': ann.get('group_id'),
            'shape_type': ann['shape_type'],
            'flags': ann.get('flags', {})
        })

    return new_annotations

def encode_image_to_base64(image_path):
    """将图像编码为base64格式"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_images_and_annotations(input_dir, output_dir, angles):
    """处理文件夹中的图像和标注文件"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            json_path = os.path.join(input_dir, filename.replace('.jpg', '.json'))

            if not os.path.exists(json_path):
                print(f"JSON file not found for {filename}, skipping...")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image = cv2.imread(image_path)

            for angle in angles:
                rotated_image = rotate_image(image, angle)
                new_image_path = os.path.join(output_dir,
                                              f"{os.path.splitext(filename)[0]}_rotated_{angle}.jpg")
                # cv2.imwrite(new_image_path, rotated_image)
                cv2.imwrite(new_image_path, rotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                rotated_annotations = rotate_annotations(data['shapes'], angle, image.shape)

                # 更新 JSON 数据中的图像尺寸和编码
                new_json_data = {
                    'version': data.get('version', '4.5.6'),
                    'flags': data.get('flags', {}),
                    'shapes': rotated_annotations,
                    'imagePath': os.path.basename(new_image_path),
                    'imageData': encode_image_to_base64(new_image_path),
                    'imageHeight': rotated_image.shape[0],
                    'imageWidth': rotated_image.shape[1]
                }

                new_json_path = os.path.join(output_dir,
                                             f"{os.path.splitext(filename)[0]}_rotated_{angle}.json")
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_data, f, indent=4, ensure_ascii=False)

                print(f"已保存旋转后的图像和标注文件：{new_image_path}, {new_json_path}")

# 示例调用
input_dir = r"D:\deep\data\scar\var\scale"  # 输入文件夹路径
# output_dir = 'D:\\python-ai\\datasets\\shuifen\\rotation_data'  # 指定输出目录
output_dir = r"D:\deep\data\scar\var\rotation"  # 指定输出目录
angles = [90, 180, 270]  # 旋转角度

process_images_and_annotations(input_dir, output_dir, angles)
