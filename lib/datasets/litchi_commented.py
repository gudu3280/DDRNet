# -*- coding: utf-8 -*-
"""
Litchi数据集处理模块
用于荔枝语义分割任务的数据加载和预处理
继承自BaseDataset基类，实现了荔枝数据集的特定处理逻辑
"""

# ------------------------------------------------------------------------------
# Litchi Dataset for DDRNet
# 荔枝数据集用于DDRNet深度双分辨率网络
# ------------------------------------------------------------------------------

import os  # 操作系统接口模块
import cv2  # OpenCV计算机视觉库
import numpy as np  # 数值计算库
from PIL import Image  # Python图像处理库

import torch  # PyTorch深度学习框架
from torch.nn import functional as F  # PyTorch函数式接口

from .base_dataset import BaseDataset  # 导入基础数据集类


class Litchi(BaseDataset):
    """
    荔枝数据集类，用于语义分割任务
    继承自BaseDataset，实现了荔枝数据集的特定加载和预处理逻辑
    """
    
    def __init__(self, 
                 root,  # 数据集根目录
                 list_path,  # 数据列表文件路径
                 num_samples=None,  # 样本数量限制，None表示使用全部样本
                 num_classes=3,  # 类别数量：背景、荔枝、荔枝茎
                 multi_scale=True,  # 是否启用多尺度训练
                 flip=True,  # 是否启用随机翻转
                 ignore_label=255,  # 忽略标签值，通常用于边界像素
                 base_size=1024,  # 基础图像尺寸
                 crop_size=(512, 512),  # 裁剪尺寸
                 downsample_rate=1,  # 下采样率
                 scale_factor=16,  # 尺度因子
                 mean=[0.485, 0.456, 0.406],  # ImageNet预训练模型的均值
                 std=[0.229, 0.224, 0.225]):  # ImageNet预训练模型的标准差
        """
        初始化荔枝数据集
        Args:
            root: 数据集根目录路径
            list_path: 包含图像和标签路径的列表文件
            num_samples: 限制使用的样本数量，用于调试或快速训练
            num_classes: 分割类别数量，荔枝数据集为3类
            multi_scale: 是否在训练时使用多尺度增强
            flip: 是否使用随机水平翻转增强
            ignore_label: 在损失计算中忽略的标签值
            base_size: 图像预处理的基础尺寸
            crop_size: 训练时的裁剪尺寸
            downsample_rate: 标签的下采样率
            scale_factor: 网络的下采样倍数
            mean: 图像归一化的均值
            std: 图像归一化的标准差
        """
        # 调用父类构造函数，初始化基础参数
        super(Litchi, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        # 设置数据集特定参数
        self.root = root  # 数据集根目录
        self.list_path = list_path  # 数据列表文件路径
        self.num_classes = num_classes  # 类别数量

        # 数据增强参数
        self.multi_scale = multi_scale  # 多尺度训练标志
        self.flip = flip  # 随机翻转标志
        
        # 读取数据列表文件，每行包含图像路径和标签路径
        self.img_list = [line.strip().split() for line in open(list_path) if line.strip()]

        # 构建文件路径列表
        self.files = self.read_files()
        
        # 如果指定了样本数量限制，则截取相应数量的样本
        if num_samples:
            self.files = self.files[:num_samples]

        # 荔枝数据集的类别名称定义
        self.class_names = ['background', 'litchi', 'litchi_stem']  # 背景、荔枝、荔枝茎

    def read_files(self):
        """
        读取并构建文件路径列表
        从数据列表文件中读取图像和标签的相对路径，
        转换为绝对路径并构建文件信息字典
        Returns:
            list: 包含文件信息字典的列表，每个字典包含img、label、name字段
        """
        files = []
        # 遍历图像列表中的每一项
        for item in self.img_list:
            # 构建图像的完整路径
            image_path = os.path.join(self.root, item[0])
            # 构建标签的完整路径
            label_path = os.path.join(self.root, item[1])
            # 提取文件名（不含扩展名）作为样本名称
            name = os.path.splitext(os.path.basename(item[0]))[0]
            # 添加到文件列表
            files.append({
                "img": image_path,    # 图像路径
                "label": label_path,  # 标签路径
                "name": name,         # 样本名称
            })
        return files

    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        实现数据加载、预处理和数据增强
        Args:
            index: 样本索引
        Returns:
            tuple: (image, label, size, name)
                - image: 预处理后的图像数组，形状为[C, H, W]
                - label: 标签数组，形状为[H, W]
                - size: 原始图像尺寸
                - name: 样本名称
        """
        # 获取指定索引的文件信息
        item = self.files[index]
        name = item["name"]
        
        # 使用OpenCV加载图像（BGR格式）
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        
        # 使用PIL加载标签图像并转换为numpy数组
        label = np.array(Image.open(item["label"]))
        
        # 记录原始图像尺寸
        size = label.shape

        # 多尺度训练：随机缩放图像和标签
        if self.multi_scale and np.random.uniform() > 0.5:
            # 随机选择缩放比例（0.5到2.0倍）
            scale = np.random.uniform(0.5, 2.0)
            # 计算缩放后的尺寸
            h, w = int(size[0] * scale), int(size[1] * scale)
            # 对图像进行双线性插值缩放
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # 对标签进行最近邻插值缩放（保持标签值不变）
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # 图像预处理：归一化和格式转换
        image = self.input_transform(image)  # 应用归一化变换
        image = image.transpose((2, 0, 1))   # 从HWC转换为CHW格式

        # 随机水平翻转增强
        if self.flip and np.random.uniform() > 0.5:
            image = image[:, :, ::-1]  # 图像水平翻转
            label = label[:, ::-1]     # 标签水平翻转

        # 随机裁剪或填充到指定尺寸
        if image.shape[1] > self.crop_size[0] or image.shape[2] > self.crop_size[1]:
            # 如果图像尺寸大于裁剪尺寸，进行随机裁剪
            h, w = image.shape[1], image.shape[2]
            
            # 计算需要填充的尺寸
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            
            # 如果需要填充，先进行填充
            if pad_h > 0 or pad_w > 0:
                # 图像用0填充
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)
                # 标签用ignore_label填充
                label = np.pad(label, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
            
            # 随机选择裁剪起始位置
            h, w = image.shape[1], image.shape[2]
            start_h = np.random.randint(0, h - self.crop_size[0] + 1)
            start_w = np.random.randint(0, w - self.crop_size[1] + 1)
            
            # 执行裁剪
            image = image[:, start_h:start_h + self.crop_size[0], start_w:start_w + self.crop_size[1]]
            label = label[start_h:start_h + self.crop_size[0], start_w:start_w + self.crop_size[1]]
        else:
            # 如果图像尺寸小于裁剪尺寸，进行填充
            h, w = image.shape[1], image.shape[2]
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                # 图像用0填充
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)
                # 标签用ignore_label填充
                label = np.pad(label, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)

        # 返回处理后的数据，使用copy()确保数据独立性
        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        """
        多尺度推理函数
        在多个尺度上进行推理并融合结果，提高分割精度
        Args:
            config: 配置对象，包含模型参数
            model: 训练好的分割模型
            image: 输入图像张量，形状为[1, C, H, W]
            scales: 推理尺度列表，默认为[1]
            flip: 是否使用翻转测试时增强
        Returns:
            torch.Tensor: 融合后的预测结果
        """
        # 获取输入图像的尺寸信息
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."  # 仅支持批次大小为1
        
        # 将张量转换为numpy数组并调整维度顺序
        image = image.numpy()[0].transpose((1,2,0)).copy()
        
        # 设置滑动窗口的步长
        stride_h = int(self.crop_size[0] * 1.0)
        stride_w = int(self.crop_size[1] * 1.0)
        
        # 初始化最终预测结果张量
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        
        # 在每个尺度上进行推理
        for scale in scales:
            # 对图像进行多尺度增强
            new_img = self.multi_scale_aug(image=image,
                                            rand_scale=scale,
                                            rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                # 小尺度：直接推理
                new_img = new_img.transpose((2, 0, 1))  # HWC -> CHW
                new_img = np.expand_dims(new_img, axis=0)  # 添加批次维度
                new_img = torch.from_numpy(new_img)  # 转换为张量
                preds = self.inference(config, model, new_img, flip)  # 模型推理
                preds = preds[:, :, 0:height, 0:width]  # 裁剪到实际尺寸
            else:
                # 大尺度：使用滑动窗口推理
                new_h, new_w = new_img.shape[:-1]
                # 计算需要的窗口数量
                rows = int(np.ceil(1.0 * new_h / stride_h))
                cols = int(np.ceil(1.0 * new_w / stride_w))
                
                # 初始化预测结果和计数器
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                # 滑动窗口推理
                for r in range(rows):
                    for c in range(cols):
                        # 计算当前窗口的位置
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        
                        # 裁剪当前窗口
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        
                        # 对当前窗口进行推理
                        pred = self.inference(config, model, crop_img, flip)
                        
                        # 累加预测结果和计数
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                
                # 计算平均预测结果
                preds = preds / count
                preds = preds[:,:,:height,:width]

            # 将预测结果上采样到原始尺寸
            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            # 累加当前尺度的预测结果
            final_pred += preds
            
        return final_pred

    def get_palette(self, n):
        """
        生成调色板用于可视化分割结果
        为每个类别生成唯一的RGB颜色
        Args:
            n: 类别数量
        Returns:
            list: 调色板列表，包含n个类别的RGB值
        """
        palette = [0] * (n * 3)  # 初始化调色板，每个类别3个RGB值
        
        # 为每个类别生成颜色
        for j in range(0, n):
            lab = j  # 当前类别标签
            # 初始化RGB值为0
            palette[j * 3 + 0] = 0  # R
            palette[j * 3 + 1] = 0  # G
            palette[j * 3 + 2] = 0  # B
            
            i = 0
            # 使用位操作生成唯一颜色
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))  # 设置R通道
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))  # 设置G通道
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))  # 设置B通道
                i += 1
                lab >>= 3  # 右移3位处理下一组位
                
        return palette

    def save_pred(self, preds, sv_path, name):
        """
        保存预测结果为彩色分割图像
        将模型的预测概率转换为类别标签，并应用调色板保存为PNG图像
        Args:
            preds: 模型预测结果，形状为[B, C, H, W]
            sv_path: 保存路径
            name: 图像名称列表
        """
        # 生成256色调色板
        palette = self.get_palette(256)
        
        # 将预测概率转换为类别标签
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        
        # 保存每个预测结果
        for i in range(preds.shape[0]):
            pred = preds[i]  # 获取第i个预测结果
            
            # 创建PIL图像对象
            save_img = Image.fromarray(pred)
            # 应用调色板
            save_img.putpalette(palette)
            # 保存为PNG文件
            save_img.save(os.path.join(sv_path, name[i]+'.png'))