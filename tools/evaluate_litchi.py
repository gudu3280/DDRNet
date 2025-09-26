# ------------------------------------------------------------------------------
# 荔枝数据集综合评估脚本
# 用于对训练好的语义分割模型在荔枝数据集上进行全面的性能评估
# 包括准确率、IoU、精确率、召回率、F1分数等指标的计算和可视化
# ------------------------------------------------------------------------------

"""
荔枝数据集模型评估脚本

主要功能：
1. 加载预训练的语义分割模型
2. 在荔枝测试数据集上进行推理
3. 计算各种评估指标（像素准确率、mIoU、精确率、召回率、F1分数等）
4. 生成详细的评估报告和可视化图表
5. 保存评估结果到JSON文件

支持的评估指标：
- 像素准确率 (Pixel Accuracy)
- 平均像素准确率 (Mean Pixel Accuracy)
- 交并比 (IoU) 和平均交并比 (mIoU)
- 精确率、召回率、F1分数
- 混淆矩阵可视化
"""

from __future__ import absolute_import  # 绝对导入
from __future__ import division         # 除法运算
from __future__ import print_function   # print函数

# 标准库导入
import argparse  # 命令行参数解析
import os        # 操作系统接口
import pprint    # 美化打印
import shutil    # 高级文件操作
import sys       # 系统相关参数和函数
import json      # JSON数据处理
import time      # 时间相关函数
import numpy as np  # 数值计算库

# 绘图相关库
import matplotlib.pyplot as plt  # 绘图库
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import seaborn as sns  # 统计数据可视化
from matplotlib import font_manager  # 字体管理

# 深度学习框架
import torch                    # PyTorch主库
import torch.nn as nn          # 神经网络模块
import torch.backends.cudnn as cudnn  # CUDNN后端
import torch.nn.functional as F  # 函数式接口
from torch.utils.data import DataLoader  # 数据加载器
from PIL import Image          # 图像处理库

# 项目特定模块
import _init_paths             # 路径初始化
import models                  # 模型定义
import datasets               # 数据集处理
from config import config     # 配置管理
from config import update_config  # 配置更新
from core.function import testval, test  # 核心测试函数
from utils.modelsummary import get_model_summary  # 模型摘要
from utils.utils import create_logger, FullModel  # 工具函数
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def compute_confusion_matrix(target, pred, num_classes):
    """
    计算混淆矩阵的简单实现，替代sklearn
    
    参数:
        target (numpy.ndarray): 真实标签数组
        pred (numpy.ndarray): 预测标签数组
        num_classes (int): 类别数量
    
    返回:
        numpy.ndarray: 混淆矩阵，形状为 (num_classes, num_classes)
    
    功能:
        计算预测结果与真实标签之间的混淆矩阵
        忽略无效标签（小于0或大于等于类别数的标签）
    """
    # 创建有效标签掩码，过滤无效标签
    mask = (target >= 0) & (target < num_classes)
    target = target[mask]  # 过滤后的真实标签
    pred = pred[mask]      # 过滤后的预测标签
    
    # 初始化混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 统计每个真实标签-预测标签对的出现次数
    for t, p in zip(target, pred):
        cm[t, p] += 1
    return cm

class LitchiEvaluator:
    """
    荔枝数据集评估器类
    
    用于计算语义分割模型在荔枝数据集上的各种评估指标
    包括像素准确率、IoU、精确率、召回率、F1分数等
    
    参数:
        num_classes (int): 类别数量，默认为3（背景、荔枝、荔枝茎）
        ignore_label (int): 忽略的标签值，默认为255
    
    主要功能:
        - 维护混淆矩阵
        - 计算像素准确率和类别像素准确率
        - 计算IoU和mIoU
        - 计算精确率、召回率和F1分数
    """
    def __init__(self, num_classes=3, ignore_label=255):
        """
        初始化评估器
        
        参数:
            num_classes (int): 类别数量
            ignore_label (int): 忽略的标签值
        """
        self.num_classes = num_classes    # 类别数量
        self.ignore_label = ignore_label  # 忽略标签
        self.reset()  # 重置混淆矩阵
        
    def reset(self):
        """
        重置混淆矩阵
        
        将混淆矩阵清零，用于开始新的评估
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def update(self, pred, target):
        """
        更新混淆矩阵
        
        参数:
            pred (numpy.ndarray): 预测标签数组
            target (numpy.ndarray): 真实标签数组
        
        功能:
            根据预测结果和真实标签更新混淆矩阵
        """
        # 计算当前批次的混淆矩阵
        cm = compute_confusion_matrix(target, pred, self.num_classes)
        # 累加到总混淆矩阵中
        self.confusion_matrix += cm
        
    def get_pixel_accuracy(self):
        """
        计算像素准确率 (Pixel Accuracy)
        
        公式: PA = (TP + TN) / (TP + TN + FP + FN)
        即对角线元素之和除以矩阵所有元素之和
        
        返回:
            float: 像素准确率
        """
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc
        
    def get_class_pixel_accuracy(self):
        """
        计算各类别的像素准确率 (Class Pixel Accuracy)
        
        公式: CPA_i = TP_i / (TP_i + FN_i)
        即每个类别的对角线元素除以该行的和
        
        返回:
            numpy.ndarray: 各类别的像素准确率数组
        """
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        class_acc = np.nan_to_num(class_acc)  # 处理除零情况
        return class_acc
        
    def get_mean_pixel_accuracy(self):
        """
        计算平均像素准确率 (Mean Pixel Accuracy, MPA)
        
        公式: MPA = (1/k) * Σ(CPA_i)
        即各类别像素准确率的平均值
        
        返回:
            float: 平均像素准确率
        """
        class_acc = self.get_class_pixel_accuracy()
        return np.mean(class_acc)
        
    def get_iou(self):
        """
        计算各类别的交并比 (Intersection over Union, IoU)
        
        公式: IoU_i = TP_i / (TP_i + FP_i + FN_i)
        即交集除以并集
        
        返回:
            numpy.ndarray: 各类别的IoU数组
        """
        # 交集：对角线元素
        intersection = np.diag(self.confusion_matrix)
        # 并集：行和 + 列和 - 交集
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / union
        iou = np.nan_to_num(iou)  # 处理除零情况
        return iou
        
    def get_miou(self):
        """
        计算平均交并比 (Mean Intersection over Union, mIoU)
        
        公式: mIoU = (1/k) * Σ(IoU_i)
        即各类别IoU的平均值
        
        返回:
            float: 平均交并比
        """
        iou = self.get_iou()
        return np.mean(iou)
        
    def get_precision_recall_fscore(self):
        """
        计算各类别的精确率、召回率和F1分数
        
        公式:
            Precision_i = TP_i / (TP_i + FP_i)  # 列方向
            Recall_i = TP_i / (TP_i + FN_i)     # 行方向
            F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
        
        返回:
            tuple: (precision, recall, fscore) 三个numpy数组
        """
        # 精确率：对角线元素除以列和
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        # 召回率：对角线元素除以行和
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        
        # 处理除零情况
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        # F1分数：精确率和召回率的调和平均
        fscore = 2 * (precision * recall) / (precision + recall)
        fscore = np.nan_to_num(fscore)  # 处理除零情况
        
        return precision, recall, fscore
        
    def get_all_metrics(self):
        """
        获取所有评估指标
        
        返回:
            dict: 包含所有评估指标的字典
                - pixel_accuracy: 像素准确率
                - class_pixel_accuracy: 各类别像素准确率
                - mean_pixel_accuracy: 平均像素准确率
                - iou: 各类别IoU
                - miou: 平均IoU
                - precision: 各类别精确率
                - recall: 各类别召回率
                - fscore: 各类别F1分数
                - confusion_matrix: 混淆矩阵
        """
        # 计算所有指标
        pixel_acc = self.get_pixel_accuracy()
        class_pixel_acc = self.get_class_pixel_accuracy()
        mpa = self.get_mean_pixel_accuracy()
        iou = self.get_iou()
        miou = self.get_miou()
        precision, recall, fscore = self.get_precision_recall_fscore()
        
        # 返回指标字典
        return {
            'pixel_accuracy': pixel_acc,
            'class_pixel_accuracy': class_pixel_acc,
            'mean_pixel_accuracy': mpa,
            'iou': iou,
            'miou': miou,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'confusion_matrix': self.confusion_matrix
        }

def create_evaluation_charts(results, output_dir, class_names):
    """
    创建综合评估图表
    
    参数:
        results (dict): 评估结果字典，包含各种指标
        output_dir (str): 输出目录路径
        class_names (list): 类别名称列表
    
    返回:
        str: 保存的图表文件路径
    
    功能:
        生成包含多个子图的综合评估图表，包括：
        1. 像素准确率指标
        2. 各类别IoU表现
        3. 精确率、召回率和F分数
        4. 模型综合性能指标
        5. 各类别详细性能对比
    """
    # 设置中文字体，确保图表中的中文正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 从结果字典中提取各项指标数据
    pixel_acc = results['pixel_accuracy']          # 像素准确率
    mean_pixel_acc = results['mean_pixel_accuracy'] # 平均像素准确率
    miou = results['miou']                         # 平均IoU
    class_pixel_acc = results['class_pixel_accuracy'] # 各类别像素准确率
    iou = results['iou']                           # 各类别IoU
    precision = results['precision']               # 各类别精确率
    recall = results['recall']                     # 各类别召回率
    fscore = results['fscore']                     # 各类别F1分数
    param_count = results['model_parameters_M']    # 模型参数量（百万）
    fps = results['fps']                           # 帧率

    # 创建大图，包含多个子图
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 像素准确率指标子图 (Pixel Accuracy Metrics)
    ax1 = plt.subplot(2, 3, 1)  # 2行3列的第1个子图
    metrics = [pixel_acc, mean_pixel_acc, miou]
    labels = ['Pixel Accuracy', 'Mean Pixel Accuracy', 'mIoU']
    colors = ['#5B9BD5', '#A5A5A5', '#FFC000']  # 蓝色、灰色、橙色
    
    # 绘制柱状图
    bars1 = ax1.bar(labels, metrics, color=colors)
    ax1.set_title('像素准确率指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_ylim(0, 1.1)  # 设置y轴范围
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars1, metrics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 各类别IoU表现子图 (Per-class IoU Performance)
    ax2 = plt.subplot(2, 3, 2)  # 2行3列的第2个子图
    # 过滤出有效的类别（IoU > 0）
    valid_classes = [i for i, name in enumerate(class_names) if iou[i] > 0]
    valid_iou = [iou[i] for i in valid_classes]
    valid_names = [class_names[i] for i in valid_classes]
    
    # 绘制有效类别的IoU柱状图
    bars2 = ax2.bar(valid_names, valid_iou, color=['#5B9BD5', '#A5A5A5'])
    ax2.set_title(f'各类别IoU表现 (mIoU = {miou:.2%})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('IoU 值', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars2, valid_iou):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 精确率、召回率和F分数子图 (Precision, Recall, F-score)
    ax3 = plt.subplot(2, 3, 3)  # 2行3列的第3个子图
    x = np.arange(len(valid_names))
    width = 0.25  # 柱子宽度
    
    # 提取有效类别的精确率、召回率和F分数
    valid_precision = [precision[i] for i in valid_classes]
    valid_recall = [recall[i] for i in valid_classes]
    valid_fscore = [fscore[i] for i in valid_classes]
    
    # 绘制分组柱状图
    bars3_1 = ax3.bar(x - width, valid_precision, width, label='精确率', color='#5B9BD5')
    bars3_2 = ax3.bar(x, valid_recall, width, label='召回率', color='#A5A5A5')
    bars3_3 = ax3.bar(x + width, valid_fscore, width, label='F分数', color='#FFC000')
    
    ax3.set_title('精确率、召回率和F分数', fontsize=14, fontweight='bold')
    ax3.set_ylabel('数值', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(valid_names)
    ax3.legend()  # 显示图例
    ax3.set_ylim(0, 1.1)
    
    # 在每个柱子上添加数值标签
    for bars in [bars3_1, bars3_2, bars3_3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 模型综合性能指标子图 (Model Performance Summary)
    ax4 = plt.subplot(2, 3, 4)  # 2行3列的第4个子图
    perf_metrics = [fps, miou, param_count]
    perf_labels = ['帧率', 'MIoU', 'Parameters(M)']
    perf_colors = ['#70AD47', '#FFC000', '#C5504B']  # 绿色、橙色、红色
    
    # 由于不同指标的数值范围差异很大，需要归一化处理以便在同一图表中显示
    normalized_fps = fps / 10  # 假设最大帧率约为10
    normalized_miou = miou     # mIoU本身就在0-1范围内
    normalized_params = param_count / 100  # 假设最大参数量约为100M
    
    normalized_values = [normalized_fps, normalized_miou, normalized_params]
    
    # 绘制归一化后的性能指标
    bars4 = ax4.bar(perf_labels, normalized_values, color=perf_colors)
    ax4.set_title('模型综合性能指标', fontsize=14, fontweight='bold')
    ax4.set_ylabel('归一化值', fontsize=12)
    ax4.set_ylim(0, 1.1)
    
    # 在柱子上显示实际数值（而非归一化值）
    actual_values = [f'{fps:.1f}', f'{miou:.1%}', f'{param_count:.1f}M']
    for bar, value in zip(bars4, actual_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                value, ha='center', va='bottom', fontsize=10)
    
    # 5. 各类别详细性能对比子图 (Detailed Per-class Performance)
    ax5 = plt.subplot(2, 1, 2)  # 占据下半部分的完整宽度
    x = np.arange(len(class_names))
    width = 0.2  # 柱子宽度
    
    # 绘制四组指标的分组柱状图
    bars5_1 = ax5.bar(x - 1.5*width, iou, width, label='IoU', color='#5B9BD5')
    bars5_2 = ax5.bar(x - 0.5*width, precision, width, label='Precision', color='#FFC000')
    bars5_3 = ax5.bar(x + 0.5*width, recall, width, label='Recall', color='#70AD47')
    bars5_4 = ax5.bar(x + 1.5*width, fscore, width, label='F-score', color='#C5504B')
    
    ax5.set_title(f'各类别详细性能对比 (mIoU = {miou:.2%})', fontsize=14, fontweight='bold')
    ax5.set_ylabel('数值', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names)
    ax5.legend()
    ax5.set_ylim(0, 1.1)
    
    # 在每个柱子上添加数值标签（只显示非零值）
    for bars in [bars5_1, bars5_2, bars5_3, bars5_4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只为非零值添加标签
                ax5.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 调整子图间距，避免重叠
    plt.tight_layout()
    
    # 保存综合图表
    chart_path = os.path.join(output_dir, 'comprehensive_evaluation_metrics.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
    plt.close()  # 关闭图形，释放内存
    
    return chart_path

def count_parameters(model):
    """
    计算模型参数数量（以百万为单位）
    
    参数:
        model (torch.nn.Module): PyTorch模型
    
    返回:
        float: 可训练参数数量（百万）
    
    功能:
        统计模型中所有需要梯度更新的参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def create_separate_charts(results, output_dir, class_names):
    """
    创建分离的评估图表
    
    参数:
        results (dict): 评估结果字典
        output_dir (str): 输出目录路径
        class_names (list): 类别名称列表
    
    返回:
        list: 保存的图表文件路径列表
    
    功能:
        为每种类型的指标创建单独的图表文件：
        1. 像素准确率指标图表
        2. IoU性能图表
        3. 精确率、召回率和F分数图表
        4. 各类别详细性能对比图表
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取评估结果数据
    pixel_acc = results['pixel_accuracy']
    mean_pixel_acc = results['mean_pixel_accuracy']
    miou = results['miou']
    class_pixel_acc = results['class_pixel_accuracy']
    iou = results['iou']
    precision = results['precision']
    recall = results['recall']
    fscore = results['fscore']
    param_count = results['model_parameters_M']
    fps = results['fps']
    
    chart_paths = []  # 存储生成的图表路径
    
    # 1. 像素准确率指标图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列布局
    
    # 左侧：整体准确率指标
    metrics = [pixel_acc, mean_pixel_acc, miou]
    labels = ['Pixel Accuracy', 'Mean Pixel Accuracy', 'mIoU']
    colors = ['#5B9BD5', '#A5A5A5', '#FFC000']
    
    bars1 = ax1.bar(labels, metrics, color=colors)
    ax1.set_title('像素准确率指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_ylim(0, 1.1)
    
    # 添加数值标签
    for bar, value in zip(bars1, metrics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 右侧：各类别像素准确率
    bars2 = ax2.bar(class_names, class_pixel_acc, color=['#5B9BD5', '#A5A5A5', '#FFC000'])
    ax2.set_title('各类别像素准确率', fontsize=14, fontweight='bold')
    ax2.set_ylabel('数值', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # 添加数值标签
    for bar, value in zip(bars2, class_pixel_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, '1_pixel_accuracy_metrics.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart_path)
    
    # 2. IoU性能图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左侧：各类别IoU
    bars1 = ax1.bar(class_names, iou, color=['#5B9BD5', '#A5A5A5', '#FFC000'])
    ax1.set_title(f'各类别IoU表现 (mIoU = {miou:.2%})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('IoU 值', fontsize=12)
    ax1.set_ylim(0, 1.1)
    
    # 只为非零值添加标签
    for bar, value in zip(bars1, iou):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 右侧：整体性能摘要
    perf_metrics = [fps, miou * 100, param_count]  # 将mIoU缩放到百分比以便可视化
    perf_labels = ['帧率', 'MIoU(%)', 'Parameters(M)']
    perf_colors = ['#70AD47', '#FFC000', '#C5504B']
    
    bars2 = ax2.bar(perf_labels, perf_metrics, color=perf_colors)
    ax2.set_title('模型综合性能指标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('数值', fontsize=12)
    
    # 添加实际数值标签
    actual_values = [f'{fps:.1f}', f'{miou:.1%}', f'{param_count:.1f}M']
    for bar, value in zip(bars2, actual_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                value, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, '2_iou_performance_metrics.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart_path)
    
    # 3. 精确率、召回率和F分数图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # 绘制三组指标的分组柱状图
    bars1 = ax.bar(x - width, precision, width, label='精确率', color='#5B9BD5')
    bars2 = ax.bar(x, recall, width, label='召回率', color='#A5A5A5')
    bars3 = ax.bar(x + width, fscore, width, label='F分数', color='#FFC000')
    
    ax.set_title('精确率、召回率和F分数', fontsize=14, fontweight='bold')
    ax.set_ylabel('数值', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # 添加数值标签（只显示非零值）
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, '3_precision_recall_fscore.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart_path)
    
    # 4. 各类别详细性能对比图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.2
    
    # 绘制四组指标的分组柱状图
    bars1 = ax.bar(x - 1.5*width, iou, width, label='IoU', color='#5B9BD5')
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#FFC000')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#70AD47')
    bars4 = ax.bar(x + 1.5*width, fscore, width, label='F-score', color='#C5504B')
    
    ax.set_title(f'各类别详细性能对比 (mIoU = {miou:.2%})', fontsize=14, fontweight='bold')
    ax.set_ylabel('数值', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # 添加数值标签（只显示非零值）
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, '4_detailed_class_performance.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart_path)
    
    return chart_paths

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的命令行参数对象
    
    功能:
        配置评估脚本的命令行参数，包括：
        - 配置文件路径
        - 模型文件路径
        - 输出目录路径
        - 其他可选配置参数
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Evaluate segmentation network')
    
    # 配置文件参数
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../experiments/litchi/ddrnet23_litchi.yaml',
                        type=str)
    
    # 模型文件参数
    parser.add_argument('--model-file',
                        help='model file path',
                        default='../output/litchi/best_val.pth',
                        type=str)
    
    # 输出目录参数
    parser.add_argument('--output-dir',
                        help='output directory',
                        default='../output/litchi/evaluation',
                        type=str)
    
    # 其他可选参数
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # 解析参数并更新配置
    args = parser.parse_args()
    update_config(config, args)
    return args

def main():
    """
    主评估函数
    
    功能:
        执行完整的模型评估流程，包括：
        1. 参数解析和配置初始化
        2. 模型加载和权重恢复
        3. 数据集准备和数据加载器创建
        4. 模型评估和指标计算
        5. 结果保存和可视化图表生成
    
    评估流程:
        - 解析命令行参数
        - 创建输出目录
        - 初始化日志记录器
        - 配置CUDNN设置
        - 构建和加载模型
        - 准备测试数据集
        - 执行评估循环
        - 计算各种评估指标
        - 生成可视化图表
        - 保存评估结果
    """
    # 1. 参数解析和配置初始化
    args = parse_args()
    
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化日志记录器
    logger, _, _ = create_logger(config, args.cfg, 'eval')
    logger.info(pprint.pformat(args))  # 打印参数信息
    logger.info(config)                # 打印配置信息

    # 2. CUDNN相关设置
    cudnn.benchmark = config.CUDNN.BENCHMARK      # 启用CUDNN基准测试
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 设置确定性模式
    cudnn.enabled = config.CUDNN.ENABLED          # 启用CUDNN

    # 3. 构建模型
    # 处理PyTorch 1.x版本的BatchNorm2d兼容性
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    
    # 创建分割模型
    model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)

    # 4. 加载模型权重
    if os.path.isfile(args.model_file):
        logger.info("=> loading checkpoint '{}'".format(args.model_file))
        # 加载检查点文件
        checkpoint = torch.load(args.model_file, map_location='cpu')
        
        # 提取状态字典
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DataParallel保存的模型（移除'module.'前缀）
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # 处理带有'model.'前缀的模型
        if any(key.startswith('model.') for key in state_dict.keys()):
            state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
        
        # 加载模型权重
        model.load_state_dict(state_dict, strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_file))
    else:
        logger.error("=> no checkpoint found at '{}'".format(args.model_file))
        return

    # 5. 计算模型参数数量
    param_count = count_parameters(model)
    logger.info(f"Model parameters: {param_count:.2f}M")

    # 6. 准备测试数据
    gpus = list(config.GPUS)  # GPU设备列表
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])  # 测试图像尺寸
    
    # 创建测试数据集
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,           # 数据集根目录
                        list_path=config.DATASET.TEST_SET,  # 测试集列表文件
                        num_samples=None,                   # 样本数量（None表示全部）
                        num_classes=config.DATASET.NUM_CLASSES,  # 类别数量
                        multi_scale=False,                  # 不使用多尺度
                        flip=False,                         # 不使用翻转
                        ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                        base_size=config.TEST.BASE_SIZE,    # 基础尺寸
                        crop_size=test_size)                # 裁剪尺寸

    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,                    # 批次大小为1
        shuffle=False,                   # 不打乱数据
        num_workers=config.WORKERS,      # 工作进程数
        pin_memory=True)                 # 启用内存锁定

    # 7. 模型设置
    model = nn.DataParallel(model, device_ids=gpus).cuda()  # 多GPU并行
    model.eval()  # 设置为评估模式

    # 8. 初始化评估器
    evaluator = LitchiEvaluator(num_classes=config.DATASET.NUM_CLASSES, 
                               ignore_label=config.TRAIN.IGNORE_LABEL)

    # 9. 评估循环
    logger.info("Starting evaluation...")
    start_time = time.time()  # 记录开始时间
    
    with torch.no_grad():  # 禁用梯度计算
        for i, (image, label, _, name) in enumerate(testloader):
            size = label.size()      # 获取标签尺寸
            image = image.cuda()     # 将图像移到GPU
            
            # 前向传播
            pred = model(image)
            # 如果输出是列表，选择指定索引的输出
            if isinstance(pred, list):
                pred = pred[config.TEST.OUTPUT_INDEX]
            
            # 将预测结果插值到原始尺寸
            pred = torch.nn.functional.interpolate(
                input=pred, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            
            # 转换为numpy数组并获取预测类别
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # 获取最大概率的类别
            
            label = label.numpy()  # 标签转为numpy数组
            
            # 更新评估器
            evaluator.update(pred.flatten(), label.flatten())
            
            # 每处理100张图像打印一次进度
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(testloader)} images")

    # 10. 获取所有评估指标
    metrics = evaluator.get_all_metrics()
    
    # 11. 计算推理时间和帧率
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(testloader)  # 平均每张图像处理时间
    fps = 1.0 / avg_time_per_image                     # 帧率
    
    # 12. 定义类别名称
    class_names = ['Background', 'Litchi', 'Litchi_Stem']
    
    # 13. 打印评估结果
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model Parameters: {param_count:.2f}M")
    logger.info(f"Average inference time: {avg_time_per_image:.4f}s")
    logger.info(f"FPS: {fps:.2f}")
    logger.info("")
    
    # 打印整体指标
    logger.info(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    logger.info(f"Mean Pixel Accuracy (MPA): {metrics['mean_pixel_accuracy']:.4f}")
    logger.info(f"Mean IoU (MIoU): {metrics['miou']:.4f}")
    logger.info("")
    
    # 打印各类别详细指标
    logger.info("Per-class metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}:")
        logger.info(f"  Class Pixel Accuracy: {metrics['class_pixel_accuracy'][i]:.4f}")
        logger.info(f"  IoU: {metrics['iou'][i]:.4f}")
        logger.info(f"  Precision: {metrics['precision'][i]:.4f}")
        logger.info(f"  Recall: {metrics['recall'][i]:.4f}")
        logger.info(f"  F-score: {metrics['fscore'][i]:.4f}")
    
    # 14. 保存结果到JSON文件
    results = {
        'model_parameters_M': float(param_count),
        'avg_inference_time_s': float(avg_time_per_image),
        'fps': float(fps),
        'pixel_accuracy': float(metrics['pixel_accuracy']),
        'mean_pixel_accuracy': float(metrics['mean_pixel_accuracy']),
        'miou': float(metrics['miou']),
        'class_names': class_names,
        'class_pixel_accuracy': metrics['class_pixel_accuracy'].tolist(),
        'iou': metrics['iou'].tolist(),
        'precision': metrics['precision'].tolist(),
        'recall': metrics['recall'].tolist(),
        'fscore': metrics['fscore'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    # 保存JSON文件
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # 15. 生成综合评估图表
    comprehensive_chart = create_evaluation_charts(results, args.output_dir, class_names)
    logger.info(f"Comprehensive chart saved to: {comprehensive_chart}")
    
    # 16. 生成分离的评估图表
    separate_charts = create_separate_charts(results, args.output_dir, class_names)
    logger.info("Separate charts saved:")
    for i, chart_path in enumerate(separate_charts, 1):
        logger.info(f"  {i}. {chart_path}")
    
    # 17. 保存混淆矩阵
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), 
            metrics['confusion_matrix'])
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()