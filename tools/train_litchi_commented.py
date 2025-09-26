# ------------------------------------------------------------------------------
# Optimized training script for Litchi dataset
# 荔枝数据集的优化训练脚本
# ------------------------------------------------------------------------------

# 导入Python 2/3兼容性模块，确保代码在不同Python版本下正常运行
from __future__ import absolute_import  # 使用绝对导入
from __future__ import division         # 除法运算返回浮点数
from __future__ import print_function   # print作为函数使用

# 导入标准库模块
import argparse    # 命令行参数解析
import os         # 操作系统接口
import pprint     # 美化打印输出
import shutil     # 高级文件操作
import sys        # 系统相关参数和函数
import logging    # 日志记录
import numpy as np # 数值计算库

# 导入PyTorch相关模块
import torch                              # PyTorch主模块
import torch.nn as nn                     # 神经网络模块
import torch.backends.cudnn as cudnn      # CUDA深度神经网络库后端
import torch.optim                        # 优化器模块
from tensorboardX import SummaryWriter    # TensorBoard可视化工具

# 导入项目自定义模块
import _init_paths                        # 初始化Python路径
import models                            # 模型定义模块
import datasets                          # 数据集处理模块
from config import config                # 配置参数
from config import update_config         # 配置更新函数
from core.criterion import CrossEntropy, OhemCrossEntropy  # 损失函数
from core.function import train, validate                  # 训练和验证函数
from utils.modelsummary import get_model_summary          # 模型摘要工具
from utils.utils import create_logger, FullModel          # 工具函数

def parse_args():
    """
    解析命令行参数的函数
    返回解析后的参数对象
    """
    # 创建参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    # 添加配置文件参数
    parser.add_argument('--cfg',
                        help='experiment configure file name',           # 实验配置文件名
                        default='experiments/litchi/ddrnet23_litchi.yaml',  # 默认配置文件路径
                        type=str)                                       # 参数类型为字符串
    
    # 添加随机种子参数，用于结果复现
    parser.add_argument('--seed', type=int, default=304)
    
    # 添加本地进程排名参数，用于分布式训练
    parser.add_argument("--local_rank", type=int, default=-1)       
    
    # 添加可选参数，用于在命令行中修改配置选项
    parser.add_argument('opts',
                        help="Modify config options using the command-line",  # 命令行修改配置选项
                        default=None,                                         # 默认值为None
                        nargs=argparse.REMAINDER)                            # 接收剩余所有参数

    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据命令行参数更新配置
    update_config(config, args)

    return args  # 返回解析后的参数

def get_sampler(dataset):
    """
    获取数据采样器的函数
    根据是否为分布式训练返回相应的采样器
    
    Args:
        dataset: 数据集对象
    
    Returns:
        采样器对象或None
    """
    # 导入分布式训练检查函数
    from utils.distributed import is_distributed
    
    # 如果是分布式训练
    if is_distributed():
        # 导入分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        # 返回分布式采样器实例
        return DistributedSampler(dataset)
    else:
        # 非分布式训练返回None
        return None

def main():
    """
    主函数：执行完整的训练流程
    """
    # 解析命令行参数
    args = parse_args()

    # 如果设置了随机种子（大于0）
    if args.seed > 0:
        import random  # 导入随机数模块
        print('Seeding with', args.seed)  # 打印种子信息
        random.seed(args.seed)            # 设置Python随机种子
        torch.manual_seed(args.seed)      # 设置PyTorch随机种子

    # 创建日志记录器和输出目录
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    
    # 修改输出目录到tub文件夹
    # 获取项目根目录下的tub文件夹路径
    tub_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tub')
    # 如果tub目录不存在则创建
    if not os.path.exists(tub_output_dir):
        os.makedirs(tub_output_dir)
    # 将最终输出目录设置为tub目录
    final_output_dir = tub_output_dir

    # 记录参数信息到日志
    logger.info(pprint.pformat(args))  # 美化打印参数
    logger.info(config)                # 打印配置信息

    # 创建TensorBoard写入器字典，用于可视化训练过程
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),  # TensorBoard写入器
        'train_global_steps': 0,              # 训练全局步数
        'valid_global_steps': 0,              # 验证全局步数
    }

    # CUDNN相关设置，用于优化GPU计算
    cudnn.benchmark = config.CUDNN.BENCHMARK      # 启用benchmark模式
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 设置确定性模式
    cudnn.enabled = config.CUDNN.ENABLED          # 启用CUDNN
    
    # 获取GPU列表
    gpus = list(config.GPUS)
    
    # 判断是否为分布式训练（local_rank >= 0表示分布式）
    distributed = args.local_rank >= 0
    
    # 如果是分布式训练
    if distributed:
        print("---------------devices:", args.local_rank)  # 打印设备信息
        device = torch.device('cuda:{}'.format(args.local_rank))  # 设置CUDA设备
        torch.cuda.set_device(device)                            # 设置当前CUDA设备
        # 初始化分布式进程组
        torch.distributed.init_process_group(
            backend="nccl",           # 使用NCCL后端
            init_method="env://",     # 使用环境变量初始化
        )        

    # 构建模型
    # 检查PyTorch版本是否以'1'开头
    if torch.__version__.startswith('1'):
        # 动态获取模型模块
        module = eval('models.'+config.MODEL.NAME)
        # 设置批归一化类
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    
    # 动态创建分割模型
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # 复制模型文件（仅在分布式训练的主进程中执行）
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)                    # 获取当前脚本目录
        models_dst_dir = os.path.join(final_output_dir, 'models')  # 目标模型目录
        # 如果目标目录存在则删除
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        # 复制模型文件到输出目录
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    # 设置批大小
    if distributed:
        # 分布式训练：每个GPU的批大小
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        # 非分布式训练：总批大小 = 每GPU批大小 × GPU数量
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # 准备训练数据
    # 设置裁剪尺寸（高度，宽度）
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    
    # 创建训练数据集
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,                    # 数据集根目录
                        list_path=config.DATASET.TRAIN_SET,          # 训练集列表文件路径
                        num_samples=None,                            # 样本数量（None表示使用全部）
                        num_classes=config.DATASET.NUM_CLASSES,     # 类别数量
                        multi_scale=config.TRAIN.MULTI_SCALE,       # 是否使用多尺度
                        flip=config.TRAIN.FLIP,                     # 是否使用翻转
                        ignore_label=config.TRAIN.IGNORE_LABEL,     # 忽略标签值
                        base_size=config.TRAIN.BASE_SIZE,           # 基础尺寸
                        crop_size=crop_size,                        # 裁剪尺寸
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE, # 下采样率
                        scale_factor=config.TRAIN.SCALE_FACTOR)     # 缩放因子

    # 获取训练数据采样器
    train_sampler = get_sampler(train_dataset)
    
    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(
        train_dataset,                                           # 训练数据集
        batch_size=batch_size,                                  # 批大小
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None, # 是否打乱（仅在非分布式时）
        num_workers=config.WORKERS,                             # 工作进程数
        pin_memory=True,                                        # 是否固定内存
        drop_last=True,                                         # 是否丢弃最后不完整的批次
        sampler=train_sampler)                                  # 数据采样器

    # 创建验证数据集
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,                # 数据集根目录
                        list_path=config.DATASET.TEST_SET,       # 测试集列表文件路径
                        num_samples=None,                        # 样本数量
                        num_classes=config.DATASET.NUM_CLASSES, # 类别数量
                        multi_scale=False,                       # 验证时不使用多尺度
                        flip=False,                              # 验证时不使用翻转
                        ignore_label=config.TRAIN.IGNORE_LABEL, # 忽略标签值
                        base_size=config.TEST.BASE_SIZE,        # 测试基础尺寸
                        crop_size=(config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]))  # 测试裁剪尺寸

    # 获取测试数据采样器
    test_sampler = get_sampler(test_dataset)
    
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(
        test_dataset,                                    # 测试数据集
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),  # 测试批大小
        shuffle=False,                                   # 测试时不打乱
        num_workers=config.WORKERS,                      # 工作进程数
        pin_memory=True,                                 # 固定内存
        sampler=test_sampler)                           # 数据采样器

    # 设置损失函数（准则）
    if config.LOSS.USE_OHEM:
        # 使用OHEM（在线困难样本挖掘）交叉熵损失
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                                     thres=config.LOSS.OHEMTHRES,             # OHEM阈值
                                     min_kept=config.LOSS.OHEMKEEP,           # 最小保留样本数
                                     weight=None)                             # 类别权重
    else:
        # 使用标准交叉熵损失
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签
                                 weight=None)                             # 类别权重

    # 将模型和损失函数封装为完整模型
    model = FullModel(model, criterion)
    
    # 根据训练模式设置模型
    if distributed:
        # 分布式训练
        model = model.cuda()                              # 将模型移到GPU
        # 使用分布式数据并行
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,                  # 查找未使用的参数
            device_ids=[args.local_rank],                # 设备ID列表
            output_device=args.local_rank                # 输出设备
        )
    else:
        # 非分布式训练
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 使用数据并行，将模型移到多个GPU
            model = nn.DataParallel(model, device_ids=gpus).cuda()
        else:
            # CUDA不可用时使用CPU训练
            print("CUDA not available, using CPU for training")
            model = nn.DataParallel(model)

    # 设置优化器
    if config.TRAIN.OPTIMIZER == 'sgd':
        # 获取模型所有参数
        params_dict = dict(model.named_parameters())
        
        # 如果配置了非骨干网络关键词（用于不同学习率）
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []      # 骨干网络参数列表
            nbb_lr = []     # 非骨干网络参数列表
            nbb_keys = set()  # 非骨干网络参数名集合
            
            # 遍历所有参数
            for k, param in params_dict.items():
                # 检查参数名是否包含非骨干网络关键词
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)    # 添加到非骨干网络参数
                    nbb_keys.add(k)         # 记录参数名
                else:
                    bb_lr.append(param)     # 添加到骨干网络参数

            print(nbb_keys)  # 打印非骨干网络参数名
            
            # 设置不同学习率的参数组
            params = [
                {'params': bb_lr, 'lr': config.TRAIN.LR},                                    # 骨干网络学习率
                {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}   # 非骨干网络学习率
            ]
        else:
            # 所有参数使用相同学习率
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        # 创建SGD优化器
        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,              # 学习率
                                momentum=config.TRAIN.MOMENTUM,  # 动量
                                weight_decay=config.TRAIN.WD,    # 权重衰减
                                nesterov=config.TRAIN.NESTEROV,  # 是否使用Nesterov动量
                                )
    else:
        # 仅支持SGD优化器
        raise ValueError('Only Support SGD optimizer')

    # 计算每个epoch的迭代次数
    epoch_iters = int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    # 初始化最佳mIoU和起始epoch
    best_mIoU = 0      # 最佳平均IoU
    last_epoch = 0     # 上次训练的epoch
    
    # 如果需要恢复训练
    if config.TRAIN.RESUME:
        # 检查点文件路径
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        # 如果检查点文件存在
        if os.path.isfile(model_state_file):
            # 加载检查点
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']      # 恢复最佳mIoU
            last_epoch = checkpoint['epoch']         # 恢复epoch
            dct = checkpoint['state_dict']           # 获取模型状态字典
            
            # 加载模型参数（处理键名前缀）
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    # 设置起始epoch
    start_epoch = last_epoch

    # 注释掉的学习率调度器代码（使用function.py中的adjust_learning_rate函数代替）
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, 
    #     lr_lambda=lambda x: (1 - x / (len(trainloader) * config.TRAIN.END_EPOCH)) ** 0.9)

    # 主训练循环
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        # 分布式训练时设置epoch（用于数据打乱）
        if distributed:
            train_sampler.set_epoch(epoch)

        # 执行训练
        train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters,
              config.TRAIN.LR, len(trainloader), trainloader, optimizer, model, writer_dict)

        # 执行验证
        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)

        # 仅在主进程中保存模型（分布式训练时local_rank <= 0表示主进程）
        if args.local_rank <= 0:
            # 记录保存检查点信息
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            
            # 保存检查点
            torch.save({
                'epoch': epoch+1,                        # 下一个epoch
                'best_mIoU': best_mIoU,                 # 最佳mIoU
                'state_dict': model.module.state_dict(), # 模型状态字典
                'optimizer': optimizer.state_dict(),     # 优化器状态字典
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            # 如果当前mIoU超过历史最佳
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU  # 更新最佳mIoU
                # 保存最佳模型
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_val.pth'))
                # 记录保存最佳模型信息
                logger.info('=> saving best model with mIoU: {:.4f}'.format(best_mIoU))
            
            # 构建日志消息
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logger.info(msg)        # 记录训练信息
            logger.info(IoU_array)  # 记录各类别IoU

    # 训练完成后保存最终模型
    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pth'))

    # 关闭TensorBoard写入器
    writer_dict['writer'].close()

# 程序入口点
if __name__ == '__main__':
    main()  # 调用主函数