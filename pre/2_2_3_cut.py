import numpy as np
import cv2
import json
import os


# 缩放图像并同步更新 JSON 数据
def resize_image(image, scale_factor=0.5, target_size=(512, 512)):
    # 计算新的图像尺寸
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # 缩放图像
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建黑色背景的图像
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 计算填充的坐标
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    # 将缩放后的图像放到黑色背景中心
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return background, x_offset, y_offset


# 更新 JSON 中的标注数据
def update_json_annotations(original_json, scale_x, scale_y, x_offset, y_offset):
    new_shapes = []
    for shape in original_json['shapes']:
        new_points = []
        for point in shape['points']:
            # 根据缩放比例更新坐标，并加入填充偏移
            new_x = (point[0] * scale_x) + x_offset
            new_y = (point[1] * scale_y) + y_offset
            new_points.append([new_x, new_y])

        if new_points:
            shape['points'] = new_points
            new_shapes.append(shape)

    original_json['imageHeight'] = 512
    original_json['imageWidth'] = 512
    original_json['shapes'] = new_shapes

    return original_json


# 直接缩放图像并更新标注
def resize_and_update_annotations(image_path, json_path, output_dir, scale_factor=0.5, target_size=(512, 512)):
    img = cv2.imread(image_path)

    # 计算缩放比例
    scale_x = scale_factor
    scale_y = scale_factor

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        original_json = json.load(f)

    # 对图片进行缩放并填充
    resized_img, x_offset, y_offset = resize_image(img, scale_factor, target_size)

    # 输出缩放后的图像坐标偏移
    print(f"Image resized. Scale: {scale_factor}, Offset: ({x_offset}, {y_offset})")

    # 更新 JSON 文件中的标注
    updated_json = update_json_annotations(original_json, scale_x, scale_y, x_offset, y_offset)

    # 输出更新后的 JSON 数据，用于调试
    print(f"Updated JSON for {os.path.basename(json_path)}:")
    print(json.dumps(updated_json, ensure_ascii=False, indent=4))

    # 保存缩放并填充后的图片
    cropped_img_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_resized.jpg")
    # cv2.imwrite(cropped_img_path, resized_img)
    cv2.imwrite(cropped_img_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # 保存更新后的 JSON 文件
    cropped_json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}_resized.json")
    with open(cropped_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(updated_json, json_file, ensure_ascii=False, indent=4)

    print(f"保存完成，缩放并填充后的图片和更新的 JSON 数据已保存在: {output_dir}")


# 批量处理文件夹中的所有图片和 JSON 文件
def process_images_in_folder(folder_path, output_dir, scale_factor=0.5, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):  # 处理常见的图片格式
            image_path = os.path.join(folder_path, file)
            json_path = os.path.join(folder_path, f"{os.path.splitext(file)[0]}.json")

            if os.path.exists(json_path):
                resize_and_update_annotations(image_path, json_path, output_dir, scale_factor, target_size)
            else:
                print(f"未找到对应的 JSON 文件: {json_path}")


# 使用方法
folder_path = r'D:\deep\data\scar\var\origin'
output_dir = r'D:\deep\data\scar\var\cut'

process_images_in_folder(folder_path, output_dir, scale_factor=0.5, target_size=(512, 512))
