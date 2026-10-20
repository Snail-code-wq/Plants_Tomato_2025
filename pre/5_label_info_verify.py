import cv2
import json
import numpy as np
from matplotlib import pyplot as plt


# 分别可视化结果
def show_image(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 设置路径
image_path = r"D:\deep\data\locule\train\scale\3_4_image_3_3_4_resized.jpg" # 替换为你的 RGB 图像路径
json_path = r"D:\deep\data\locule\train\scale\3_4_image_3_3_4_resized.json" # 替换为你的 JSON 标注文件路径

# 读取 RGB 图像
image = cv2.imread(image_path)

# 读取 JSON 文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历标注信息
for shape in data['shapes']:
    points = np.array(shape['points'], dtype=np.int32)
    label = shape['label']

    # 绘制多边形
    cv2.polylines(image, [points], isClosed=True, color=(0, 100, 255), thickness=2)

    # 在多边形上添加标签
    centroid = points.mean(axis=0).astype(int)
    # cv2.putText(image, label, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


show_image(image, 'Image with Annotations')
# # 显示图像
# cv2.imshow('Image with Annotations', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()