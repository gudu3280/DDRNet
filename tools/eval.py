# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""
模型评估脚本
用于评估训练好的语义分割模型在测试集上的性能
计算mIoU、像素准确率、平均准确率等指标
"""

# 标准库导入
import argparse  # 命令行参数解析
import os        # 操作系统接口
import pprint    # 美化打印输出
import shutil    # 高级文件操作
import sys       # 系统相关参数和函数

# 日志和时间相关
import logging   # 日志记录
import time      # 时间处理
import timeit    # 精确计时
from pathlib import Path  # 路径操作

# 数值计算
import numpy as np  # 数值计算库

# 深度学习框架
import torch                        # PyTorch主库
import torch.nn as nn              # 神经网络模块
import torch.backends.cudnn as cudnn  # CUDNN后端优化

# 项目特定模块
import _init_paths  # 初始化路径设置
import models       # 模型定义
import datasets     # 数据集处理
from config import config          # 配置管理
from config import update_config   # 配置更新
from core.function import testval, test  # 核心测试函数
from utils.modelsummary import get_model_summary  # 模型摘要工具
from utils.utils import create_logger, FullModel, speed_test  # 实用工具函数

def parse_args():
    """
    解析命令行参数
    
    功能：
    - 定义和解析命令行参数
    - 设置默认配置文件路径
    - 更新全局配置
    
    返回：
    - args: 解析后的命令行参数对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    # 配置文件参数
    parser.add_argument('--cfg',
                        help='experiment configure file name',  # 实验配置文件名
                        default="experiments/cityscapes/ddrnet23_slim.yaml",  # 默认配置文件
                        type=str)
    
    # 其他配置选项参数
    parser.add_argument('opts',
                        help="Modify config options using the command-line",  # 通过命令行修改配置选项
                        default=None,
                        nargs=argparse.REMAINDER)  # 接收剩余的所有参数

    # 解析参数
    args = parser.parse_args()
    
    # 更新全局配置
    update_config(config, args)

    return args

def main():
    """
    主函数 - 执行模型评估流程
    
    功能：
    - 解析命令行参数和配置
    - 初始化日志系统
    - 构建和加载模型
    - 准备测试数据集
    - 执行模型评估
    - 计算和输出性能指标
    """
    # 解析命令行参数
    args = parse_args()

    # 创建日志记录器和输出目录
    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    # 记录参数和配置信息
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # CUDNN相关设置 - 优化GPU计算性能
    cudnn.benchmark = config.CUDNN.BENCHMARK      # 启用基准测试模式
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 设置确定性计算
    cudnn.enabled = config.CUDNN.ENABLED          # 启用CUDNN

    # 构建模型
    if torch.__version__.startswith('1'):
        # PyTorch 1.x版本的兼容性处理
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    
    # 根据配置创建分割模型
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # 可选：打印模型摘要信息（已注释）
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # 确定模型文件路径
    if config.TEST.MODEL_FILE:
        # 使用配置中指定的模型文件
        model_state_file = config.TEST.MODEL_FILE
    else:
        # 使用输出目录中的最佳模型
        model_state_file = os.path.join(final_output_dir, 'best.pth')      
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')      
    
    logger.info('=> loading model from {}'.format(model_state_file))
        
    # 加载预训练模型权重
    pretrained_dict = torch.load(model_state_file)
    
    # 处理不同格式的权重文件
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    
    # 获取当前模型的状态字典
    model_dict = model.state_dict()
    
    # 过滤和匹配权重参数（去除前缀'module.'）
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    
    # 记录加载的参数
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> 从预训练模型加载参数层 {}'.format(k))
    
    # 更新模型参数并加载权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # ==================== 模型并行化设置 ====================
    # 设置多GPU并行计算
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # ==================== 测试数据准备 ====================
    # 设置测试图像尺寸
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])  # 测试图像尺寸
    
    # 创建测试数据集
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,              # 数据集根目录
                        list_path=config.DATASET.TEST_SET,     # 测试集列表文件
                        num_samples=None,                      # 样本数量（None表示全部）
                        num_classes=config.DATASET.NUM_CLASSES,  # 类别数量
                        multi_scale=False,                     # 不使用多尺度测试
                        flip=False,                           # 不使用翻转增强
                        ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签值
                        base_size=config.TEST.BASE_SIZE,       # 基础尺寸
                        crop_size=test_size,                   # 裁剪尺寸
                        downsample_rate=1)                     # 下采样率

    # 创建数据加载器
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,                    # 批次大小为1（测试时通常使用单张图片）
        shuffle=False,                   # 不打乱数据顺序
        num_workers=config.WORKERS,      # 数据加载工作进程数
        pin_memory=True)                 # 固定内存以加速GPU数据传输
    
    # ==================== 执行模型评估 ====================
    # 开始计时
    start = timeit.default_timer()
    
    # 执行模型评估，计算各种性能指标
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                        test_dataset,   # 测试数据集
                                                        testloader,     # 数据加载器
                                                        model,          # 评估模型
                                                        sv_pred=False)  # 不保存预测结果图像

    # ==================== 输出评估结果 ====================
    # 格式化并输出主要评估指标
    msg = '平均IoU: {: 4.4f}, 像素准确率: {: 4.4f}, \
        平均准确率: {: 4.4f}, 各类别IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)  # 输出每个类别的IoU数组

    # 结束计时并输出总运行时间
    end = timeit.default_timer()
    logger.info('总耗时: %d 分钟' % np.int((end-start)/60))
    logger.info('评估完成')


if __name__ == '__main__':
    main()
