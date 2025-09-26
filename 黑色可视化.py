import cv2
import numpy as np
import os

labels_dir = 'data2\\labels'  # 修改为data2目录
visual_dir = 'data2\\visual_labels'

os.makedirs(visual_dir, exist_ok=True)

label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]

for label_file in label_files:
    label_path = os.path.join(labels_dir, label_file)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    visual_img =np.zeros_like(label_img, dtype=np.uint8)
    visual_img[label_img == 1] = 128
    visual_img[label_img == 2] = 255

    visual_path = os.path.join(visual_dir, label_file)
    cv2.imwrite(visual_path, visual_img)
    print(f"创建可视化图片: {visual_path}")
print("所用可视化图片已经创建成功！")