# -*- coding: utf-8 -*-
"""
工具函数模块
包含训练、评估和可视化过程中使用的各种辅助函数
提供模型封装、指标计算、日志记录、学习率调整等功能
"""

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# 微软版权声明，MIT许可证
# 作者：Ke Sun
# ------------------------------------------------------------------------------

from __future__ import absolute_import  # 绝对导入
from __future__ import division  # 除法运算
from __future__ import print_function  # 打印函数

import os  # 操作系统接口
import logging  # 日志记录模块
import time  # 时间处理模块
from pathlib import Path  # 路径处理模块

import numpy as np  # 数值计算库

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口


class FullModel(nn.Module):
    """
    完整模型封装类
    用于在多GPU环境下分布式计算损失，减少主GPU的内存消耗
    将模型和损失函数封装在一起，便于并行训练
    
    参考讨论：
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    
    def __init__(self, model, loss):
        """
        初始化完整模型
        Args:
            model: 分割网络模型
            loss: 损失函数
        """
        super(FullModel, self).__init__()
        self.model = model  # 分割模型
        self.loss = loss    # 损失函数

    def pixel_acc(self, pred, label):
        """
        计算像素级准确率
        Args:
            pred: 模型预测结果，形状为[B, C, H, W]
            label: 真实标签，形状为[B, H, W]
        Returns:
            torch.Tensor: 像素准确率
        """
        # 如果预测结果和标签尺寸不匹配，进行双线性插值调整
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        
        # 获取预测类别（取概率最大的类别）
        _, preds = torch.max(pred, dim=1)
        
        # 创建有效像素掩码（排除ignore_label）
        valid = (label >= 0).long()
        
        # 计算正确预测的像素数量
        acc_sum = torch.sum(valid * (preds == label).long())
        
        # 计算总的有效像素数量
        pixel_sum = torch.sum(valid)
        
        # 计算准确率，添加小常数避免除零
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        """
        前向传播函数
        Args:
            inputs: 输入图像
            labels: 真实标签
            *args, **kwargs: 其他参数
        Returns:
            tuple: (损失值, 模型输出, 像素准确率)
        """
        # 模型前向传播
        outputs = self.model(inputs, *args, **kwargs)
        
        # 计算损失
        loss = self.loss(outputs, labels)
        
        # 计算像素准确率（使用主要输出）
        acc = self.pixel_acc(outputs[1], labels)
        
        # 返回损失（增加维度）、输出和准确率
        return torch.unsqueeze(loss, 0), outputs, acc


class AverageMeter(object):
    """
    平均值计算器
    用于计算和存储训练过程中各种指标的当前值和平均值
    常用于记录损失、准确率等训练指标
    """

    def __init__(self):
        """初始化平均值计算器"""
        self.initialized = False  # 是否已初始化标志
        self.val = None          # 当前值
        self.avg = None          # 平均值
        self.sum = None          # 累计和
        self.count = None        # 计数

    def initialize(self, val, weight):
        """
        初始化计算器
        Args:
            val: 初始值
            weight: 权重
        """
        self.val = val           # 设置当前值
        self.avg = val           # 初始平均值等于当前值
        self.sum = val * weight  # 初始累计和
        self.count = weight      # 初始计数
        self.initialized = True  # 标记为已初始化

    def update(self, val, weight=1):
        """
        更新计算器
        Args:
            val: 新的值
            weight: 权重，默认为1
        """
        if not self.initialized:
            # 如果未初始化，先进行初始化
            self.initialize(val, weight)
        else:
            # 如果已初始化，添加新值
            self.add(val, weight)

    def add(self, val, weight):
        """
        添加新值到计算器
        Args:
            val: 新值
            weight: 权重
        """
        self.val = val                    # 更新当前值
        self.sum += val * weight          # 累加到总和
        self.count += weight              # 增加计数
        self.avg = self.sum / self.count  # 重新计算平均值

    def value(self):
        """返回当前值"""
        return self.val

    def average(self):
        """返回平均值"""
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    """
    创建日志记录器
    设置文件日志和控制台日志，创建输出目录和TensorBoard日志目录
    Args:
        cfg: 配置对象，包含输出目录等设置
        cfg_name: 配置文件名称
        phase: 训练阶段，默认为'train'
    Returns:
        tuple: (logger对象, 输出目录路径, TensorBoard日志目录路径)
    """
    # 创建根输出目录路径对象
    root_output_dir = Path(cfg.OUTPUT_DIR)
    
    # 如果根输出目录不存在，创建它
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    # 获取数据集名称和模型名称
    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    
    # 提取配置文件名（不含扩展名）
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    # 创建最终输出目录：根目录/数据集/配置名
    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    # 创建目录，如果父目录不存在也会创建，如果已存在则不报错
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳字符串
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    # 创建日志文件名：配置名_时间戳_阶段.log
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    
    # 设置日志格式
    head = '%(asctime)-15s %(message)s'
    
    # 配置文件日志
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    
    # 获取logger对象并设置级别
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 添加控制台输出处理器
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # 创建TensorBoard日志目录
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    计算混淆矩阵
    用于评估分割模型的性能，统计各类别的预测准确情况
    Args:
        label: 真实标签张量
        pred: 预测结果张量
        size: 图像尺寸
        num_class: 类别数量
        ignore: 忽略的标签值，默认为-1
    Returns:
        numpy.ndarray: 混淆矩阵，形状为[num_class, num_class]
    """
    # 将预测结果转换为numpy数组并调整维度顺序
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 获取预测类别（取概率最大的类别）
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    
    # 将真实标签转换为numpy数组并裁剪到指定尺寸
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    # 创建忽略索引掩码（排除ignore标签）
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    # 计算线性索引：真实类别 * 类别数 + 预测类别
    index = (seg_gt * num_class + seg_pred).astype('int32')
    
    # 统计每个索引的出现次数
    label_count = np.bincount(index)
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_class, num_class))

    # 填充混淆矩阵
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
                
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters, 
                        cur_iters, power=0.9, nbb_mult=10):
    """
    调整学习率
    使用多项式衰减策略调整学习率
    Args:
        optimizer: 优化器对象
        base_lr: 基础学习率
        max_iters: 最大迭代次数
        cur_iters: 当前迭代次数
        power: 衰减指数，默认0.9
        nbb_mult: 第二个参数组的学习率倍数，默认10
    Returns:
        float: 调整后的学习率
    """
    # 确保所有参数都是实数类型
    base_lr = float(base_lr)
    max_iters = float(max_iters)
    cur_iters = float(cur_iters)
    power = float(power)
    nbb_mult = float(nbb_mult)
    
    # 使用多项式衰减公式计算学习率
    lr = base_lr * ((1 - cur_iters / max_iters) ** power)
    
    # 确保结果是实数（处理可能的复数结果）
    lr = float(lr.real) if hasattr(lr, 'real') else float(lr)
    
    # 设置第一个参数组的学习率
    optimizer.param_groups[0]['lr'] = lr
    
    # 如果有第二个参数组，设置其学习率（通常用于不同的网络部分）
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
        
    return lr


# 导入额外的库用于图像处理和可视化
import cv2  # OpenCV计算机视觉库
from PIL import Image  # Python图像处理库


def colorEncode(labelmap, colors, mode='RGB'):
    """
    将标签图转换为彩色图像
    为每个类别分配特定颜色，用于可视化分割结果
    Args:
        labelmap: 标签图，每个像素值代表类别
        colors: 颜色映射表，每个类别对应一个RGB颜色
        mode: 颜色模式，'RGB'或'BGR'
    Returns:
        numpy.ndarray: 彩色编码后的图像
    """
    # 确保标签图为整数类型
    labelmap = labelmap.astype('int')
    
    # 初始化RGB图像
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    
    # 为每个唯一标签分配颜色
    for label in np.unique(labelmap):
        if label < 0:  # 跳过负标签（通常是ignore标签）
            continue
        # 为当前标签的所有像素分配对应颜色
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    # 根据模式返回RGB或BGR格式
    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]  # 反转颜色通道顺序
    else:
        return labelmap_rgb


class Vedio(object):
    """
    视频处理类
    用于创建和保存视频文件，将图像序列合成为视频
    """
    
    def __init__(self, video_path):
        """
        初始化视频处理器
        Args:
            video_path: 视频文件保存路径
        """
        self.video_path = video_path
        # 创建视频写入器：mp4v编码，15fps，分辨率1280x480
        self.cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 480))

    def addImage(self, img, colorMask):
        """
        向视频中添加图像帧
        将原图和彩色掩码水平拼接后添加到视频
        Args:
            img: 原始图像
            colorMask: 彩色分割掩码
        """
        # 将RGB转换为BGR（OpenCV使用BGR格式）
        img = img[:, :, ::-1]
        colorMask = colorMask[:, :, ::-1]
        
        # 水平拼接原图和掩码
        img = np.concatenate([img, colorMask], axis=1)
        
        # 写入视频帧
        self.cap.write(img)

    def releaseCap(self):
        """释放视频写入器资源"""
        self.cap.release()


class Map16(object):
    """
    16类室内场景分割可视化类
    用于室内场景的语义分割结果可视化和分析
    """
    
    def __init__(self, vedioCap, visualpoint=True):
        """
        初始化Map16可视化器
        Args:
            vedioCap: 视频捕获对象
            visualpoint: 是否可视化特征点
        """
        # 定义16个室内场景类别名称
        self.names = ("background", "floor", "bed", "cabinet,wardrobe,bookcase,shelf",
                "person", "door", "table,desk,coffee", "chair,armchair,sofa,bench,swivel,stool",
                "rug", "railing", "column", "refrigerator", "stairs,stairway,step", "escalator", "wall",
                "dog", "plant")
        
        # 定义每个类别对应的RGB颜色
        self.colors = np.array([[0, 0, 0],        # 背景-黑色
                    [0, 0, 255],      # 地板-蓝色
                    [0, 255, 0],      # 床-绿色
                    [0, 255, 255],    # 柜子-青色
                    [255, 0, 0],      # 人-红色
                    [255, 0, 255],    # 门-品红色
                    [255, 255, 0],    # 桌子-黄色
                    [255, 255, 255],  # 椅子-白色
                    [0, 0, 128],      # 地毯-深蓝色
                    [0, 128, 0],      # 栏杆-深绿色
                    [128, 0, 0],      # 柱子-深红色
                    [0, 128, 128],    # 冰箱-深青色
                    [128, 0, 0],      # 楼梯-深红色
                    [128, 0, 128],    # 扶梯-深品红色
                    [128, 128, 0],    # 墙-橄榄色
                    [128, 128, 128],  # 狗-灰色
                    [192, 192, 192]], dtype=np.uint8)  # 植物-浅灰色
        
        self.outDir = "output/map16"  # 输出目录
        self.vedioCap = vedioCap      # 视频捕获对象
        self.visualpoint = visualpoint # 是否显示特征点
    
    def visualize_result(self, data, pred, dir, img_name=None):
        """
        可视化分割结果
        生成彩色分割图像，统计类别分布，可选显示特征点
        Args:
            data: 原始图像数据
            pred: 预测结果
            dir: 保存目录
            img_name: 图像名称
        """
        img = data
        pred = np.int32(pred)  # 确保预测结果为整数类型
        
        # 统计各类别的像素数量和占比
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        
        # 按像素数量降序排列并打印占比超过0.1%的类别
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx]]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        # 可选：计算和显示特征点
        if self.visualpoint:
            img = img.copy()
            # 转换为灰度图像用于特征点检测
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = np.float32(img_gray)
            
            # 使用goodFeaturesToTrack检测角点
            goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, 400, 0.01, 10)
            goodfeatures_corners = np.int0(goodfeatures_corners)
            
            # 在图像上绘制特征点
            for i in goodfeatures_corners:
                x, y = i.flatten()  # 展平坐标
                cv2.circle(img, (x, y), 3, [0, 255], -1)  # 绘制绿色圆点

        # 将预测结果转换为彩色图像
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        # 创建可视化图像：原图70% + 彩色掩码30%
        im_vis = img * 0.7 + pred_color * 0.3
        im_vis = im_vis.astype(np.uint8)

        # 添加到视频
        self.vedioCap.addImage(im_vis, pred_color)

        # 保存可视化结果
        img_name = img_name
        if not os.path.exists(dir):
            os.makedirs(dir)  # 创建保存目录
        
        # 保存为PNG图像
        Image.fromarray(im_vis).save(
            os.path.join(dir, img_name))


def speed_test(model, size=896, iteration=100):
    """
    模型速度测试函数
    测试模型的推理速度和FPS性能
    Args:
        model: 待测试的模型
        size: 输入图像尺寸，默认896
        iteration: 测试迭代次数，默认100
    """
    # 创建随机输入张量
    input_t = torch.Tensor(1, 3, size, size).cuda()
    feed_dict = {}
    feed_dict['img_data'] = input_t

    print("start warm up")

    # 预热阶段：运行10次预热GPU
    for i in range(10):
        model(feed_dict, segSize=(size, size))

    print("warm up done")
    
    # 记录开始时间
    start_ts = time.time()
    
    # 正式测试：运行指定次数的推理
    for i in range(iteration):
        model(feed_dict, segSize=(size, size))

    # 同步CUDA操作，确保所有GPU操作完成
    torch.cuda.synchronize()
    
    # 记录结束时间
    end_ts = time.time()

    # 计算总耗时
    t_cnt = end_ts - start_ts
    
    # 打印性能结果
    print("=======================================")
    print("FPS: %f" % (100 / t_cnt))                    # 每秒帧数
    print(f"Inference time {t_cnt/100*1000} ms")        # 单次推理时间（毫秒）