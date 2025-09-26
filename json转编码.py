# convert_labelme_to_mask.py
import os
import json
import numpy as np
import cv2
from labelme import utils

# 定义路径 - 修改为data2目录
annotations_dir = 'data2\\images'  # JSON文件夹路径
images_dir = 'data2\\images'            # 原图文件夹路径
labels_dir = 'data2\\labels'            # 输出标签文件夹路径

# 创建输出目录
os.makedirs(labels_dir, exist_ok=True)

# 类别名称到数值的映射字典
class_name_to_id = {"_background_": 0, "lychee": 1, "stem": 2} # 背景很重要，必须为0

# 遍历所有json文件
for json_file in os.listdir(annotations_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(annotations_dir, json_file)
        
        # 读取json文件
        data = json.load(open(json_path))
        
        # 获取图片尺寸
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # 创建一个全零的数组作为初始标签图
        label_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # 处理json中的每一个形状（多边形）
        for shape in data['shapes']:
            class_name = shape['label']
            points = shape['points']
            shape_type = shape.get('shape_type', 'polygon')
            
            # 获取类别ID
            class_id = class_name_to_id[class_name]
            
            # 将点转换为NumPy数组
            points = np.array(points, dtype=np.int32)
            
            # 在label_mask上绘制多边形，填充值为类别ID
            if shape_type == 'polygon':
                cv2.fillPoly(label_mask, [points], color=class_id)
            # 如果你的标注还有点、线，可以在这里添加其他处理逻辑
            # elif shape_type == 'line':
            #   cv2.polylines(...)
            # else: ...
        
        # 生成输出文件名（与json同名，但扩展名为.png）
        base_name = os.path.splitext(json_file)[0] # 例如 '1.json' -> '1'
        output_path = os.path.join(labels_dir, base_name + '.png')
        
        # 保存生成的标签图
        cv2.imwrite(output_path, label_mask)
        print(f"Saved: {output_path}")

print("All conversions completed!")