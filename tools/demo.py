# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""
模型演示脚本
功能说明：
- 用于演示训练好的语义分割模型的推理能力
- 加载预训练模型并在测试数据集上进行推理
- 显示模型结构信息和推理性能统计
- 保存推理结果到指定目录

使用方法：
python tools/demo.py --cfg experiments/litchi/ddrnet23_litchi.yaml
"""

# ==================== 标准库导入 ====================
import argparse    # 命令行参数解析
import os          # 操作系统接口
import pprint      # 美化打印输出
import shutil      # 高级文件操作
import sys         # 系统相关参数和函数

import logging     # 日志记录
import time        # 时间相关函数
import timeit      # 精确计时工具
from pathlib import Path  # 面向对象的文件路径操作

import numpy as np # 数值计算库

# ==================== 深度学习库导入 ====================
import torch                           # PyTorch深度学习框架
import torch.nn as nn                  # 神经网络模块
import torch.backends.cudnn as cudnn   # CUDA深度神经网络库后端

# ==================== 项目模块导入 ====================
import _init_paths                     # 初始化Python路径
import models                          # 模型定义模块
import datasets                        # 数据集处理模块
from config import config              # 配置管理
from config import update_config       # 配置更新函数
from core.function import testval, test # 测试和验证函数
from utils.modelsummary import get_model_summary  # 模型结构摘要工具
from utils.utils import create_logger, FullModel, speed_test  # 工具函数

def parse_args():
    """
    解析命令行参数函数
    
    功能说明：
    - 定义并解析演示脚本的命令行参数
    - 设置配置文件路径和其他可选参数
    - 更新全局配置对象
    
    返回值：
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='训练语义分割网络')
    
    # 配置文件路径参数
    parser.add_argument('--cfg',
                        help='实验配置文件名称',
                        default="experiments/map/map_hrnet_ocr_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml",
                        type=str)
    
    # 其他配置选项参数
    parser.add_argument('opts',
                        help="使用命令行修改配置选项",
                        default=None,
                        nargs=argparse.REMAINDER)

    # 解析参数并更新配置
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    """
    主函数 - 执行模型演示的核心流程
    
    功能说明：
    - 解析命令行参数并初始化日志系统
    - 构建和加载预训练的语义分割模型
    - 准备测试数据集和数据加载器
    - 执行模型推理并记录性能统计
    
    处理流程：
        1. 参数解析和日志初始化
        2. CUDNN设置和模型构建
        3. 预训练权重加载
        4. 数据集准备和推理执行
        5. 性能统计和结果保存
    """
    # ==================== 第一步：参数解析和日志初始化 ====================
    args = parse_args()

    # 创建日志记录器和输出目录
    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    # 记录参数和配置信息
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # ==================== 第二步：CUDNN设置和模型构建 ====================
    # 配置CUDNN相关设置以优化性能
    cudnn.benchmark = config.CUDNN.BENCHMARK      # 启用基准测试模式
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 确定性计算
    cudnn.enabled = config.CUDNN.ENABLED          # 启用CUDNN

    # 构建语义分割模型
    if torch.__version__.startswith('1'):
        # PyTorch 1.x版本的兼容性处理
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    
    # 根据配置创建分割模型
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # ==================== 第三步：模型结构分析和权重加载 ====================
    # 创建随机输入张量用于模型结构分析
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    # 记录模型结构摘要信息，包括参数量、计算量等
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # 确定模型权重文件路径
    if config.TEST.MODEL_FILE:
        # 使用配置文件中指定的模型路径
        model_state_file = config.TEST.MODEL_FILE
    else:
        # 使用默认的最佳模型路径
        # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
        model_state_file = os.path.join(final_output_dir, 'best.pth')    
    logger.info('=> 从 {} 加载模型权重'.format(model_state_file))
        
    # 加载预训练模型权重
    pretrained_dict = torch.load(model_state_file)
    # 如果权重文件包含'state_dict'键，则提取其值
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    
    # 获取当前模型的状态字典
    model_dict = model.state_dict()
    
    # 过滤预训练权重，只保留与当前模型匹配的参数
    # 去除键名前缀'module.'（通常来自DataParallel包装）
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    
    # 记录成功加载的参数层
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> 从预训练模型加载参数层 {}'.format(k))
    
    # 更新模型参数并加载权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # ==================== 第四步：模型并行化设置 ====================
    # 获取GPU设备列表并设置数据并行
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # ==================== 第五步：测试数据准备 ====================
    # 设置测试图像尺寸
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    
    # 创建测试数据集
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,                    # 数据集根目录
                        list_path=config.DATASET.TEST_SET,           # 测试集列表文件
                        num_samples=None,                            # 样本数量（None表示使用全部）
                        num_classes=config.DATASET.NUM_CLASSES,     # 分割类别数
                        multi_scale=False,                           # 不使用多尺度测试
                        flip=False,                                  # 不使用翻转增强
                        ignore_label=config.TRAIN.IGNORE_LABEL,     # 忽略标签值
                        base_size=config.TEST.BASE_SIZE,             # 基础尺寸
                        crop_size=test_size,                         # 裁剪尺寸
                        downsample_rate=1)                           # 下采样率

    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,                    # 测试时使用批大小为1
        shuffle=False,                   # 不打乱数据顺序
        num_workers=config.WORKERS,      # 数据加载进程数
        pin_memory=True)                 # 将数据加载到固定内存

    # ==================== 第六步：执行测试推理 ====================
    # 记录开始时间
    start = timeit.default_timer()

    # 执行测试推理，保存结果到指定目录
    test(config, 
            test_dataset,                              # 测试数据集
            testloader,                                # 测试数据加载器
            model,                                     # 训练好的模型
            sv_dir=final_output_dir+'/test_result')    # 结果保存目录

    # 记录结束时间并计算总耗时
    end = timeit.default_timer()
    logger.info('总耗时: %d 分钟' % np.int((end-start)/60))
    logger.info('演示完成')


if __name__ == '__main__':
    main()
