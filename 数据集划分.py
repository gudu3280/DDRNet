import os
import random

image_dir = 'data2\\images'  # 修改为data2目录
mask_dir = 'data2\\labels'
output_dir = 'data2\\list'  # 输出到data2/list目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg','.jpeg', '.png'))]

valid_files = []
for img_file in image_files:
    mask_file = os.path.splitext(img_file)[0] + '.png'
    mask_path = os.path.join(mask_dir,mask_file)
    if os.path.exists(mask_path):
        valid_files.append(img_file)
    else:
        print(f"警告: 找不到掩码文件 {mask_file} 对应的图像 {img_file}")
    
random.shuffle(valid_files)

split_idx = int(0.8 * len(valid_files))
train_files = valid_files[:split_idx]
val_files = valid_files[split_idx:]

# 修改为.lst扩展名
with open(os.path.join(output_dir, 'train.lst'), 'w') as f:
    for img_file in train_files:
        mask_file = os.path.splitext(img_file)[0] + '.png'
        f.write(f"{os.path.join(image_dir,img_file)} {os.path.join(mask_dir, mask_file)}\n")

with open(os.path.join(output_dir, 'val.lst'), 'w') as f:
    for img_file in val_files:
        mask_file = os.path.splitext(img_file)[0] + '.png'
        f.write(f"{os.path.join(image_dir,img_file)} {os.path.join(mask_dir, mask_file)}\n")

print(f"已创建训练集列表: {len(train_files)} 个样本")
print(f"已创建验证集列表: {len(val_files)} 个样本")
print("文件已保存为 train.lst 和 val.lst")