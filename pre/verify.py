import matplotlib.pyplot as plt
from PIL import Image

# 加载图像和标注
image = Image.open(r"D:\deep\data\scar\cityscapes_dataset-scar\leftImg8bit\test\3_4_image_1_1_1.jpg")
label = Image.open(r"D:\deep\data\scar\cityscapes_dataset-scar\gtFine\test\3_4_image_1_1_1_gtFine_labelIds.png")

# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Semantic Segmentation')
plt.imshow(label, cmap='jet')
plt.axis('off')
plt.show()
