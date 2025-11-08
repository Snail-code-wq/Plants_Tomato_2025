import os

import cv2
import numpy as np


# 
def extract_tomato(image_path):
    # 
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    # red
    lower_red1 = np.array([0, 75, 75])
    upper_red1 = np.array([190, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # black
    # lower_red1 = np.array([0, 42, 20])
    # upper_red1 = np.array([200, 255, 255])
    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([179, 255, 255])

    # 
    lower_red1 = np.array([0, 60, 0])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])


    # yello
    # lower_red1 = np.array([0, 50, 0])
    # upper_red1 = np.array([37, 255, 255])
    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([179, 255, 255])

    # orange
    # lower_red1 = np.array([0, 65, 0])
    # upper_red1 = np.array([25, 255, 255])
    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([179, 255, 255])

    # 创建红色掩码
    mask_red1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # 定义果梗洼的范围
    lower_stem_scar = np.array([8, 50, 70])
    upper_stem_scar = np.array([50, 200, 255])

    # 创建果梗洼掩码
    mask_stem_scar = cv2.inRange(imgHSV, lower_stem_scar, upper_stem_scar)

    # 去除噪音
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    # 找到红色区域的轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建最终掩码
    final_mask = np.zeros_like(mask_stem_scar)

    # 计算番茄轮廓的中心点
    tomato_centers = []
    for cnt in contours_red:
        if cv2.contourArea(cnt) > 1000:  # 只保留较大的区域
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                tomato_centers.append((cX, cY))
                cv2.drawContours(final_mask, [cnt], -1, (255), thickness=cv2.FILLED)

    # 转换成 NumPy 数组以提高效率
    mask_stem_scar_indices = np.argwhere(mask_stem_scar > 0)

    # 保留靠近番茄的果梗洼
    for center in tomato_centers:
        distance_threshold = 200  # 可根据需要调整
        distances = np.sqrt(
            (mask_stem_scar_indices[:, 1] - center[0]) ** 2 + (mask_stem_scar_indices[:, 0] - center[1]) ** 2)
        close_indices = mask_stem_scar_indices[distances < distance_threshold]

        for y, x in close_indices:
            final_mask[y, x] = 255

    return img, final_mask


# 函数：根据掩码绘制番茄的最外部边缘
def draw_outer_edges(mask):
    # 进行闭运算以填补小孔
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 提取边缘
    edges = cv2.Canny(mask_closed, 100, 200)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)

    outer_edges = np.zeros_like(mask)
    contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小轮廓
    min_area = 50  # 根据需要调整最小面积
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(outer_edges, [contour], -1, (255), thickness=1)

    return outer_edges, contours


def extract_and_place_on_black_background(original_img, outer_edges):
    # 创建黑色背景
    black_background = np.zeros_like(original_img)

    # 找到外部轮廓
    contours, _ = cv2.findContours(outer_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建掩码
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)

    # 填充掩码中的区域
    cv2.fillPoly(mask, contours, 255)

    # 将原图像中对应区域放入黑色背景
    black_background[mask > 0] = original_img[mask > 0]

    return black_background


# 批量处理文件夹中的图像
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            original_img, mask = extract_tomato(image_path)
            outer_edges, contours = draw_outer_edges(mask)
            final_image = extract_and_place_on_black_background(original_img, outer_edges)

            # 保存处理后的图片
            output_final_image_path = os.path.join(output_folder, filename)

            cv2.imwrite(output_final_image_path, final_image)

# 示例用法
input_folder = r'D:\deep\data\scar'
output_folder = r"D:\deep\data\scar_cut"

process_folder(input_folder, output_folder)


