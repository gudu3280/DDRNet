# ------------------------------------------------------------------------------
# Single Image Inference Script for Litchi Segmentation
# 单张图片推理脚本 - 荔枝分割
# ------------------------------------------------------------------------------

import argparse
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import _init_paths
import models
from config import config
from config import update_config

class LitchiInference:
    def __init__(self, config_file, model_file):
        """
        初始化推理器
        Args:
            config_file: 配置文件路径
            model_file: 模型权重文件路径
        """
        # 更新配置
        update_config(config, argparse.Namespace(cfg=config_file, opts=[]))
        
        # 设置CUDNN
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        
        # 构建模型
        if torch.__version__.startswith('1'):
            module = eval('models.' + config.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        
        self.model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
        
        # 加载模型权重
        self.load_model(model_file)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.input_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])  # (height, width)
        
        # 类别信息
        self.class_names = ['Background', 'Litchi', 'Litchi_Stem']
        self.class_colors = [
            [0, 0, 0],      # 背景 - 黑色
            [255, 0, 0],    # 荔枝 - 红色
            [0, 255, 0]     # 荔枝梗 - 绿色
        ]
        
        print(f"✅ 模型加载成功！")
        print(f"📱 设备: {self.device}")
        print(f"🖼️ 输入尺寸: {self.input_size}")
        print(f"🏷️ 类别: {self.class_names}")
    
    def load_model(self, model_file):
        """加载模型权重"""
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        print(f"📂 加载模型: {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DataParallel保存的模型
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 处理模型键名不匹配的问题
        if list(state_dict.keys())[0].startswith('model.'):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def preprocess_image(self, image_path):
        """
        图像预处理
        Args:
            image_path: 图像路径
        Returns:
            tensor: 预处理后的图像张量
            original_size: 原始图像尺寸 (height, width)
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_size = image.shape[:2]  # (height, width)
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]), 
                          interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # 转换为张量
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)  # 添加batch维度
        
        return image, original_size
    
    def postprocess_prediction(self, pred, original_size):
        """
        后处理预测结果
        Args:
            pred: 模型预测输出
            original_size: 原始图像尺寸
        Returns:
            mask: 分割掩码 (numpy array)
            colored_mask: 彩色分割掩码
        """
        # 如果模型有多个输出，取第一个
        if isinstance(pred, list):
            pred = pred[0]
        
        # 调整到原始尺寸
        pred = F.interpolate(pred, size=original_size, 
                           mode='bilinear', align_corners=False)
        
        # 转换为numpy并获取类别预测
        pred = pred.cpu().numpy()[0]  # 移除batch维度
        mask = np.argmax(pred, axis=0)  # 获取类别索引
        
        # 创建彩色掩码
        colored_mask = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(self.class_colors):
            colored_mask[mask == class_id] = color
        
        return mask, colored_mask
    
    def inference(self, image_path, save_result=True, output_dir='inference_results'):
        """
        对单张图片进行推理
        Args:
            image_path: 输入图像路径
            save_result: 是否保存结果
            output_dir: 输出目录
        Returns:
            dict: 包含推理结果的字典
        """
        print(f"🔍 开始推理: {image_path}")
        
        # 预处理
        start_time = time.time()
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        preprocess_time = time.time() - start_time
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            pred = self.model(image_tensor)
        inference_time = time.time() - start_time
        
        # 后处理
        start_time = time.time()
        mask, colored_mask = self.postprocess_prediction(pred, original_size)
        postprocess_time = time.time() - start_time
        
        # 统计各类别像素数量
        unique, counts = np.unique(mask, return_counts=True)
        class_stats = {}
        total_pixels = mask.size
        
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                percentage = (count / total_pixels) * 100
                class_stats[class_name] = {
                    'pixels': int(count),
                    'percentage': round(percentage, 2)
                }
        
        # 保存结果
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存分割掩码
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8))
            
            # 保存彩色掩码
            colored_path = os.path.join(output_dir, f"{base_name}_colored.png")
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(colored_path, colored_mask_bgr)
            
            # 保存叠加图像
            original_image = cv2.imread(image_path)
            overlay = cv2.addWeighted(original_image, 0.7, colored_mask_bgr, 0.3, 0)
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            
            print(f"💾 结果已保存到: {output_dir}")
            print(f"   - 分割掩码: {mask_path}")
            print(f"   - 彩色掩码: {colored_path}")
            print(f"   - 叠加图像: {overlay_path}")
        
        # 打印统计信息
        print(f"⏱️ 推理时间:")
        print(f"   - 预处理: {preprocess_time:.4f}s")
        print(f"   - 模型推理: {inference_time:.4f}s")
        print(f"   - 后处理: {postprocess_time:.4f}s")
        print(f"   - 总时间: {preprocess_time + inference_time + postprocess_time:.4f}s")
        
        print(f"📊 分割统计:")
        for class_name, stats in class_stats.items():
            print(f"   - {class_name}: {stats['pixels']} 像素 ({stats['percentage']}%)")
        
        return {
            'mask': mask,
            'colored_mask': colored_mask,
            'class_stats': class_stats,
            'timing': {
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
        }

def parse_args():
    parser = argparse.ArgumentParser(description='单张图片荔枝分割推理')
    
    parser.add_argument('--image', '-i',
                        required=True,
                        help='输入图像路径')
    
    parser.add_argument('--config', '-c',
                        default='experiments/litchi/ddrnet23_litchi.yaml',
                        help='配置文件路径')
    
    parser.add_argument('--model', '-m',
                        default='pth/best_val.pth',
                        help='模型权重文件路径')
    
    parser.add_argument('--output', '-o',
                        default='inference_results',
                        help='输出目录')
    
    parser.add_argument('--no-save',
                        action='store_true',
                        help='不保存结果文件')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 荔枝分割推理工具")
    print("=" * 50)
    
    try:
        # 创建推理器
        inferencer = LitchiInference(args.config, args.model)
        
        # 执行推理
        result = inferencer.inference(
            image_path=args.image,
            save_result=not args.no_save,
            output_dir=args.output
        )
        
        print("✅ 推理完成！")
        
    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())