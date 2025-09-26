# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""
模型训练脚本
用于训练语义分割模型，支持分布式训练
包含完整的训练流程：数据加载、模型构建、优化器设置、训练循环、验证和模型保存
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
import torch.optim                 # 优化器模块
from tensorboardX import SummaryWriter  # TensorBoard日志记录

# 项目特定模块
import _init_paths  # 初始化路径设置
import models       # 模型定义
import datasets     # 数据集处理
from config import config          # 配置管理
from config import update_config   # 配置更新
from core.criterion import CrossEntropy, OhemCrossEntropy  # 损失函数
from core.function import train, validate  # 核心训练和验证函数
from utils.modelsummary import get_model_summary  # 模型摘要工具
from utils.utils import create_logger, FullModel  # 实用工具函数

def parse_args():
    """
    解析命令行参数
    
    功能：
    - 定义和解析命令行参数
    - 设置默认配置文件路径
    - 支持分布式训练参数
    - 更新全局配置
    
    返回：
    - args: 解析后的命令行参数对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    # 配置文件参数
    parser.add_argument('--cfg',
                        help='experiment configure file name',  # 实验配置文件名
                        default="experiments/cityscapes/ddrnet_slim.yaml",  # 默认配置文件
                        type=str)
    
    # 随机种子参数
    parser.add_argument('--seed', type=int, default=304)  # 随机种子，用于结果复现
    
    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1)  # 本地GPU排名，-1表示非分布式
    
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

def get_sampler(dataset):
    """
    获取数据采样器
    
    功能：
    - 根据是否分布式训练返回相应的采样器
    - 分布式训练时使用DistributedSampler
    - 非分布式训练时返回None
    
    参数：
    - dataset: 数据集对象
    
    返回：
    - sampler: 数据采样器或None
    """
    from utils.distributed import is_distributed
    if is_distributed():
        # 分布式训练时使用分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        # 非分布式训练时返回None
        return None

def main():
    """
    主函数 - 执行完整的模型训练流程
    
    功能：
    - 解析命令行参数和配置
    - 设置随机种子确保结果可复现
    - 初始化日志系统和TensorBoard
    - 配置分布式训练环境
    - 构建和配置模型
    - 准备训练和测试数据集
    - 设置损失函数和优化器
    - 执行训练循环和验证
    - 保存模型检查点和最终模型
    """
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子以确保结果可复现
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)        # 设置Python随机种子
        torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    # 创建日志记录器、输出目录和TensorBoard日志目录
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    # 记录参数和配置信息
    logger.info(pprint.pformat(args))
    logger.info(config)

    # 初始化TensorBoard写入器字典
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),  # TensorBoard写入器
        'train_global_steps': 0,              # 训练全局步数
        'valid_global_steps': 0,              # 验证全局步数
    }

    # CUDNN相关设置 - 优化GPU计算性能
    cudnn.benchmark = config.CUDNN.BENCHMARK      # 启用基准测试模式
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 设置确定性计算
    cudnn.enabled = config.CUDNN.ENABLED          # 启用CUDNN
    
    # 获取GPU列表和分布式训练设置
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0  # 判断是否为分布式训练
    
    if distributed:
        # 分布式训练初始化
        print("---------------devices:", args.local_rank)
        device = torch.device('cuda:{}'.format(args.local_rank))  # 设置当前设备
        torch.cuda.set_device(device)  # 设置CUDA设备
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",  # 初始化进程组
        )        

    # 构建模型
    if torch.__version__.startswith('1'):
        # PyTorch 1.x版本的兼容性处理
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    
    # 根据配置创建分割模型
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # 可选：打印模型摘要信息（已注释）
    # dump_input = torch.rand( (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]) )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # 复制模型文件到输出目录（仅在分布式训练的主进程中执行）
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)  # 删除已存在的目录
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)  # 复制模型文件

    # 设置批次大小
    if distributed:
        # 分布式训练：每个GPU的批次大小
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        # 单机多GPU训练：总批次大小
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # 准备训练数据
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])  # 裁剪尺寸
    
    # 创建训练数据集
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,              # 数据集根目录
                        list_path=config.DATASET.TRAIN_SET,    # 训练集列表文件
                        num_samples=None,                      # 样本数量（None表示全部）
                        num_classes=config.DATASET.NUM_CLASSES,  # 类别数量
                        multi_scale=config.TRAIN.MULTI_SCALE,  # 多尺度训练
                        flip=config.TRAIN.FLIP,               # 翻转增强
                        ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                        base_size=config.TRAIN.BASE_SIZE,      # 基础尺寸
                        crop_size=crop_size,                   # 裁剪尺寸
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,  # 下采样率
                        scale_factor=config.TRAIN.SCALE_FACTOR)  # 缩放因子

    # 获取训练数据采样器
    train_sampler = get_sampler(train_dataset)
    
    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,                                    # 批次大小
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,   # 是否打乱（分布式时由采样器控制）
        num_workers=config.WORKERS,                               # 工作进程数
        pin_memory=True,                                          # 固定内存
        drop_last=True,                                           # 丢弃最后不完整的批次
        sampler=train_sampler)                                    # 数据采样器

    # 初始化额外训练轮次迭代数
    extra_epoch_iters = 0
    
    # 如果配置了额外训练集
    if config.DATASET.EXTRA_TRAIN_SET:
        # 创建额外训练数据集
        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,  # 额外训练集列表文件
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)
        
        # 获取额外训练数据采样器
        extra_train_sampler = get_sampler(extra_train_dataset)
        
        # 创建额外训练数据加载器
        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)
        
        # 计算额外训练轮次的迭代数
        extra_epoch_iters = np.int(extra_train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    # 准备测试数据
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])  # 测试图像尺寸
    
    # 创建测试数据集
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,              # 数据集根目录
                        list_path=config.DATASET.TEST_SET,     # 测试集列表文件
                        num_samples=config.TEST.NUM_SAMPLES,   # 测试样本数量
                        num_classes=config.DATASET.NUM_CLASSES,  # 类别数量
                        multi_scale=False,                     # 不使用多尺度
                        flip=False,                           # 不使用翻转
                        ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                        base_size=config.TEST.BASE_SIZE,       # 基础尺寸
                        crop_size=test_size,                   # 裁剪尺寸
                        downsample_rate=1)                     # 下采样率为1

    # 获取测试数据采样器
    test_sampler = get_sampler(test_dataset)
    
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,                    # 测试批次大小为1
        shuffle=False,                   # 不打乱测试数据
        num_workers=config.WORKERS,      # 工作进程数
        pin_memory=True,                 # 固定内存
        sampler=test_sampler)            # 数据采样器

    # 设置损失函数
    if config.LOSS.USE_OHEM:
        # 使用OHEM（在线困难样本挖掘）交叉熵损失
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                                        thres=config.LOSS.OHEMTHRES,          # OHEM阈值
                                        min_kept=config.LOSS.OHEMKEEP,        # 最小保留样本数
                                        weight=train_dataset.class_weights)   # 类别权重
    else:
        # 使用标准交叉熵损失
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                                    weight=train_dataset.class_weights)   # 类别权重

    # 将模型和损失函数封装为完整模型
    model = FullModel(model, criterion)
    
    # 设置模型的并行计算
    if distributed:
        # 分布式训练设置
        model = model.to(device)  # 将模型移动到指定设备
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,      # 查找未使用的参数
            device_ids=[args.local_rank],     # 设备ID列表
            output_device=args.local_rank     # 输出设备
        )
    else:
        # 单机多GPU并行训练
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    
    # 设置优化器
    if config.TRAIN.OPTIMIZER == 'sgd':
        # 获取模型参数字典
        params_dict = dict(model.named_parameters())
        
        # 如果配置了非骨干网络关键词，使用不同的学习率
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []      # 骨干网络参数
            nbb_lr = []     # 非骨干网络参数
            nbb_keys = set()  # 非骨干网络参数名集合
            
            # 分离骨干网络和非骨干网络参数
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)  # 非骨干网络参数
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)   # 骨干网络参数
            
            print(nbb_keys)  # 打印非骨干网络参数名
            
            # 设置不同学习率的参数组
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, 
                     {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            # 所有参数使用相同学习率
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        # 创建SGD优化器
        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,              # 学习率
                                momentum=config.TRAIN.MOMENTUM,   # 动量
                                weight_decay=config.TRAIN.WD,     # 权重衰减
                                nesterov=config.TRAIN.NESTEROV,   # 是否使用Nesterov动量
                                )
    else:
        raise ValueError('Only Support SGD optimizer')  # 仅支持SGD优化器

    # 计算每个epoch的迭代数
    epoch_iters = np.int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    # 初始化最佳mIoU和起始epoch
    best_mIoU = 0
    last_epoch = 0
    
    # 如果配置了恢复训练
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')  # 检查点文件路径
        if os.path.isfile(model_state_file):
            # 加载检查点
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']    # 最佳mIoU
            last_epoch = checkpoint['epoch']       # 上次训练的epoch
            dct = checkpoint['state_dict']         # 模型状态字典
            
            # 加载模型权重（处理键名前缀）
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        
        # 分布式训练时同步所有进程
        if distributed:
            torch.distributed.barrier()

    # 开始训练计时
    start = timeit.default_timer()
    
    # 计算总训练轮次和迭代数
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH  # 总轮次
    num_iters = config.TRAIN.END_EPOCH * epoch_iters               # 常规训练迭代数
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters     # 额外训练迭代数
    
    # 训练循环
    for epoch in range(last_epoch, end_epoch):

        # 选择当前轮次使用的数据加载器
        current_trainloader = extra_trainloader if epoch >= config.TRAIN.END_EPOCH else trainloader
        
        # 为分布式采样器设置epoch（确保每个epoch的数据打乱不同）
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        # 可选：每个epoch前进行验证（已注释）
        # valid_loss, mean_IoU, IoU_array = validate(config, 
        #         testloader, model, writer_dict)

        # 执行训练
        if epoch >= config.TRAIN.END_EPOCH:
            # 额外训练阶段
            train(config, epoch-config.TRAIN.END_EPOCH, 
                  config.TRAIN.EXTRA_EPOCH, extra_epoch_iters, 
                  config.TRAIN.EXTRA_LR, extra_iters, 
                  extra_trainloader, optimizer, model, writer_dict)
        else:
            # 常规训练阶段
            train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        # 每10个epoch进行一次验证
        if epoch % 10 == 0:
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)

        # 保存模型（仅在主进程中执行）
        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            
            # 保存检查点
            torch.save({
                'epoch': epoch+1,                        # 下一个epoch
                'best_mIoU': best_mIoU,                 # 最佳mIoU
                'state_dict': model.module.state_dict(), # 模型状态字典
                'optimizer': optimizer.state_dict(),     # 优化器状态字典
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
            
            # 如果当前mIoU更好，保存最佳模型
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best.pth'))
            
            # 记录训练结果
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)  # 记录每个类别的IoU

    # 训练完成后的清理工作（仅在主进程中执行）
    if args.local_rank <= 0:
        # 保存最终模型状态
        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_state.pth'))

        # 关闭TensorBoard写入器
        writer_dict['writer'].close()
        
        # 结束计时并记录总训练时间
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end-start)/3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
