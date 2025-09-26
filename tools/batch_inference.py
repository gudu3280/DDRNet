#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图片推理脚本
功能说明：
- 对指定文件夹中的所有图片进行语义分割推理
- 支持多种图片格式（jpg, jpeg, png, bmp, tiff）
- 自动生成三种输出结果：原始掩码、彩色掩码、叠加图像
- 支持GPU和CPU推理
- 自动创建带时间戳的输出目录，避免文件覆盖
- 显示推理进度和性能统计

使用方法：
python tools/batch_inference.py --input-dir data1/images
"""

# ==================== 标准库导入 ====================
import os          # 操作系统接口，用于文件路径操作
import sys         # 系统相关参数和函数
import argparse    # 命令行参数解析
import time        # 时间相关函数，用于计算推理耗时
import glob        # 文件路径模式匹配
from pathlib import Path      # 面向对象的文件路径操作
from datetime import datetime # 日期时间处理，用于生成时间戳

# ==================== 深度学习库导入 ====================
import torch                           # PyTorch深度学习框架
import torch.nn as nn                  # 神经网络模块
import torch.backends.cudnn as cudnn   # CUDA深度神经网络库后端
import numpy as np                     # 数值计算库
from PIL import Image                  # Python图像处理库
import cv2                            # OpenCV计算机视觉库

# ==================== 项目路径设置 ====================
# 获取项目根目录路径（当前文件的上两级目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python模块搜索路径
sys.path.insert(0, project_root)
# 将lib目录添加到Python模块搜索路径
sys.path.insert(0, os.path.join(project_root, 'lib'))

# ==================== 项目模块导入 ====================
from lib.config import config          # 配置管理模块
from lib.config import update_config   # 配置更新函数
from lib.utils.utils import create_logger  # 日志创建工具
from lib.models import get_seg_model    # 语义分割模型获取函数


def parse_args():
    """
    解析命令行参数
    
    功能说明：
    - 定义并解析脚本运行时的命令行参数
    - 设置各参数的默认值和帮助信息
    - 更新全局配置对象
    
    返回值：
        args: 解析后的参数对象，包含所有命令行参数
    """
    parser = argparse.ArgumentParser(description='批量图片推理')
    
    # 配置文件路径参数
    parser.add_argument('--cfg',
                        help='配置文件路径，包含模型结构和训练参数',
                        default='experiments/litchi/ddrnet23_litchi.yaml',
                        type=str)
    
    # 模型权重文件路径参数
    parser.add_argument('--model-path',
                        help='预训练模型权重文件路径',
                        default='pth/best_val.pth',
                        type=str)
    
    # 输入图片目录参数
    parser.add_argument('--input-dir',
                        help='包含待推理图片的输入目录路径',
                        default='data1/images',
                        type=str)
    # 输出结果目录参数
    parser.add_argument('--output-dir',
                        help='推理结果保存目录，如不指定则自动生成带时间戳的目录',
                        default=None,
                        type=str)
    
    # 推理设备参数
    parser.add_argument('--device',
                        help='推理使用的计算设备（cuda使用GPU，cpu使用CPU）',
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        type=str)
    
    # 其他配置选项参数
    parser.add_argument('opts',
                        help="修改配置文件中的选项，格式：KEY1 VALUE1 KEY2 VALUE2",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用解析的参数更新全局配置对象
    update_config(config, args)
    
    return args


def preprocess_image(image_path, input_size=(1024, 2048)):
    """
    图像预处理函数
    
    功能说明：
    - 读取并预处理输入图像，使其符合模型输入要求
    - 执行颜色空间转换、尺寸调整、归一化等操作
    - 将图像转换为PyTorch张量格式
    
    参数：
        image_path (str): 输入图像的文件路径
        input_size (tuple): 模型期望的输入尺寸 (height, width)，默认(1024, 2048)
    
    返回值：
        tensor (torch.Tensor): 预处理后的图像张量，形状为(1, 3, H, W)
        original_size (tuple): 原始图像尺寸 (height, width)
    
    异常：
        ValueError: 当无法读取图像文件时抛出
    """
    # 使用OpenCV读取图像文件
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 记录原始图像尺寸，用于后续结果调整
    original_size = image.shape[:2]  # (height, width)
    
    # 将图像从BGR颜色空间转换为RGB颜色空间
    # OpenCV默认使用BGR，而深度学习模型通常使用RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整图像尺寸到模型输入要求
    # 使用双线性插值进行尺寸调整
    image = cv2.resize(image, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # 图像归一化处理
    # 使用ImageNet数据集的均值和标准差进行归一化
    mean = np.array([0.485, 0.456, 0.406])  # RGB通道均值
    std = np.array([0.229, 0.224, 0.225])   # RGB通道标准差
    image = image.astype(np.float32) / 255.0  # 将像素值从[0,255]缩放到[0,1]
    image = (image - mean) / std              # 标准化处理
    
    # 转换为PyTorch张量格式
    # 从HWC格式转换为CHW格式，并添加batch维度
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return image, original_size


def postprocess_prediction(prediction, original_size, num_classes=2):
    """
    后处理预测结果函数
    
    功能说明：
    - 将模型的原始预测输出转换为可用的分割掩码
    - 处理不同格式的模型输出（列表、元组、张量）
    - 将预测结果调整回原始图像尺寸
    
    参数：
        prediction: 模型的原始预测输出，可能是张量、列表或元组
        original_size (tuple): 原始图像尺寸 (height, width)
        num_classes (int): 分割类别数量，默认为2（背景+荔枝）
    
    返回值：
        result (numpy.ndarray): 处理后的分割掩码，像素值为类别索引
    
    异常：
        ValueError: 当预测输出不是张量格式时抛出
    """
    # 处理不同格式的模型输出
    if isinstance(prediction, (list, tuple)):
        # 如果输出是列表或元组，通常取最后一个元素作为主要预测结果
        # 这是因为某些模型会输出多个尺度的预测结果
        prediction = prediction[-1]
    
    # 确保预测结果是PyTorch张量格式
    if not isinstance(prediction, torch.Tensor):
        raise ValueError(f"预测输出必须是张量，但得到: {type(prediction)}")
    
    # 获取预测类别
    # 使用argmax函数在通道维度上找到概率最大的类别
    pred = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    # 将预测结果调整回原始图像尺寸
    # 使用最近邻插值保持类别标签的离散性
    pred = cv2.resize(pred.astype(np.uint8), 
                     (original_size[1], original_size[0]), 
                     interpolation=cv2.INTER_NEAREST)
    
    return pred


def create_colored_mask(mask, num_classes=2):
    """
    创建彩色分割掩码函数
    
    功能说明：
    - 将灰度分割掩码转换为彩色可视化图像
    - 为不同的分割类别分配不同的颜色
    - 便于直观查看分割结果
    
    参数：
        mask (numpy.ndarray): 灰度分割掩码，像素值为类别索引
        num_classes (int): 分割类别数量，默认为2
    
    返回值：
        colored_mask (numpy.ndarray): 彩色掩码图像，BGR格式
    """
    # 定义类别颜色映射表（BGR格式，适配OpenCV）
    color_map = {
        0: [0, 0, 0],        # 背景类别 - 黑色
        1: [0, 255, 0],      # 荔枝类别 - 绿色
    }
    
    # 创建空的彩色掩码图像
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 为每个类别分配对应颜色
    for class_id, color in color_map.items():
        if class_id < num_classes:
            # 将属于当前类别的像素设置为对应颜色
            colored_mask[mask == class_id] = color
    
    return colored_mask


def batch_inference(args):
    """
    批量推理主函数
    
    功能说明：
    - 执行批量图像语义分割推理的核心函数
    - 加载预训练模型并对指定目录中的所有图像进行推理
    - 生成三种类型的输出：原始掩码、彩色掩码、叠加图像
    - 显示推理进度和性能统计信息
    
    参数：
        args: 命令行参数对象，包含配置文件路径、模型路径、输入输出目录等
    
    处理流程：
        1. 设备设置和输出目录创建
        2. 模型加载和权重初始化
        3. 图像文件搜索和批量处理
        4. 结果保存和性能统计
    """
    
    # ==================== 第一步：设备设置和目录准备 ====================
    # 根据参数和硬件可用性选择计算设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录 - 如果没有指定，则使用带时间戳的目录名避免覆盖
    if args.output_dir is None:
        # 生成时间戳格式：YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"tub/batch_inference_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    # 创建主输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 创建三个子目录用于保存不同类型的结果
    mask_dir = output_dir / 'masks'      # 原始二值掩码
    colored_dir = output_dir / 'colored' # 彩色可视化掩码
    overlay_dir = output_dir / 'overlay' # 原图与掩码叠加图像
    
    # 确保所有子目录存在
    mask_dir.mkdir(exist_ok=True)
    colored_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    
    # 构建模型
    print("正在加载模型...")
    model = get_seg_model(config)
    
    # 加载预训练权重
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的权重文件格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理键名不匹配的问题
    model_dict = model.state_dict()
    
    # 如果权重文件中的键名有前缀，尝试去除前缀
    if any(key.startswith('model.') for key in state_dict.keys()):
        # 去除 'model.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    # 只加载匹配的权重
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_dict and value.shape == model_dict[key].shape:
            filtered_state_dict[key] = value
        else:
            print(f"跳过不匹配的权重: {key}")
    
    # 加载权重
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"成功加载 {len(filtered_state_dict)}/{len(model_dict)} 个权重参数")
    
    # 设置为评估模式
    model.to(device)
    model.eval()
    
    print("模型加载完成!")
    
    # 获取所有图片文件
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始批量推理...")
    
    # 批量推理
    start_time = time.time()
    
    with torch.no_grad():
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"正在处理 ({i}/{len(image_files)}): {image_path.name}")
                
                # 预处理
                input_tensor, original_size = preprocess_image(str(image_path))
                input_tensor = input_tensor.to(device)
                
                # 推理
                prediction = model(input_tensor)
                
                # 后处理
                mask = postprocess_prediction(prediction, original_size, config.DATASET.NUM_CLASSES)
                
                # 保存结果
                base_name = image_path.stem
                
                # 1. 保存原始掩码
                mask_path = mask_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), mask * 255)  # 将0,1转换为0,255
                
                # 2. 保存彩色掩码
                colored_mask = create_colored_mask(mask, config.DATASET.NUM_CLASSES)
                colored_path = colored_dir / f"{base_name}_colored.png"
                cv2.imwrite(str(colored_path), colored_mask)
                
                # 3. 保存叠加图像
                original_image = cv2.imread(str(image_path))
                if original_image is not None:
                    # 调整原图尺寸以匹配掩码
                    original_resized = cv2.resize(original_image, (original_size[1], original_size[0]))
                    
                    # 创建叠加图像
                    overlay = cv2.addWeighted(original_resized, 0.7, colored_mask, 0.3, 0)
                    overlay_path = overlay_dir / f"{base_name}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                
                # 显示进度
                if i % 10 == 0 or i == len(image_files):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (len(image_files) - i) * avg_time
                    print(f"进度: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%), "
                          f"平均耗时: {avg_time:.2f}s/张, 预计剩余: {remaining:.1f}s")
                
            except Exception as e:
                print(f"处理图片 {image_path.name} 时出错: {str(e)}")
                continue
    
    total_time = time.time() - start_time
    print(f"\n批量推理完成!")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均耗时: {total_time/len(image_files):.2f}s/张")
    print(f"结果保存在: {output_dir}")
    print(f"  - 原始掩码: {mask_dir}")
    print(f"  - 彩色掩码: {colored_dir}")
    print(f"  - 叠加图像: {overlay_dir}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    
    # 执行批量推理
    batch_inference(args)


if __name__ == '__main__':
    main()