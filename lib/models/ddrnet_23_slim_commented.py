# -*- coding: utf-8 -*-
"""
DDRNet-23-slim 模型定义文件
Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation
这是DDRNet的轻量级版本，用于实时语义分割任务
"""

import math  # 数学运算库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from torch.nn import init  # 权重初始化模块
from collections import OrderedDict  # 有序字典

# 批归一化层定义，使用PyTorch内置的BatchNorm2d
BatchNorm2d = nn.BatchNorm2d
# 批归一化动量参数，控制移动平均的更新速度
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3卷积层构造函数，带填充
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数  
        stride: 卷积步长，默认为1
    Returns:
        nn.Conv2d: 3x3卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    基础残差块，用于构建ResNet的基本单元
    expansion = 1 表示输出通道数不扩展
    """
    expansion = 1  # 通道扩展倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        """
        初始化基础残差块
        Args:
            inplanes: 输入通道数
            planes: 中间层通道数
            stride: 步长
            downsample: 下采样层，用于匹配残差连接的维度
            no_relu: 是否在最后不使用ReLU激活
        """
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 第一个批归一化层
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        # ReLU激活函数，inplace=True节省内存
        self.relu = nn.ReLU(inplace=True)
        # 第二个3x3卷积层
        self.conv2 = conv3x3(planes, planes)
        # 第二个批归一化层
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        # 下采样层，用于残差连接
        self.downsample = downsample
        # 步长参数
        self.stride = stride
        # 是否在输出时不使用ReLU
        self.no_relu = no_relu

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图
        Returns:
            输出特征图
        """
        # 保存输入作为残差连接
        residual = x

        # 第一个卷积-批归一化-激活序列
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积-批归一化序列
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：输出 = 卷积结果 + 残差
        out += residual

        # 根据no_relu参数决定是否使用ReLU激活
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    """
    瓶颈残差块，用于更深的网络结构
    expansion = 2 表示输出通道数是中间层的2倍
    """
    expansion = 2  # 通道扩展倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        """
        初始化瓶颈残差块
        Args:
            inplanes: 输入通道数
            planes: 中间层通道数
            stride: 步长
            downsample: 下采样层
            no_relu: 是否在最后不使用ReLU激活，默认为True
        """
        super(Bottleneck, self).__init__()
        # 1x1卷积，降维
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        # 3x3卷积，主要计算
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        # 1x1卷积，升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 下采样层
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图
        Returns:
            输出特征图
        """
        # 保存输入作为残差连接
        residual = x

        # 1x1卷积降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3卷积主要计算
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1卷积升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下采样，对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接
        out += residual
        
        # 根据no_relu参数决定是否使用ReLU激活
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Module):
    """
    Deep Aggregation Pyramid Pooling Module (DAPPM)
    深度聚合金字塔池化模块，用于捕获多尺度上下文信息
    """
    def __init__(self, inplanes, branch_planes, outplanes):
        """
        初始化DAPPM模块
        Args:
            inplanes: 输入通道数
            branch_planes: 分支通道数
            outplanes: 输出通道数
        """
        super(DAPPM, self).__init__()
        
        # 尺度1：5x5平均池化，步长2
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),  # 平均池化
            BatchNorm2d(inplanes, momentum=bn_mom),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),  # 1x1卷积
        )
        
        # 尺度2：9x9平均池化，步长4
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        # 尺度3：17x17平均池化，步长8
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        # 尺度4：自适应全局平均池化
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应池化到1x1
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        # 尺度0：原始尺度处理
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        # 处理模块1：对尺度1的结果进行进一步处理
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        
        # 处理模块2：对尺度2的结果进行进一步处理
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        
        # 处理模块3：对尺度3的结果进行进一步处理
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        
        # 处理模块4：对尺度4的结果进行进一步处理
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        
        # 压缩模块：将5个分支的特征图合并并压缩到输出通道数
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),  # 5个分支的通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        
        # 快捷连接：直接将输入映射到输出通道数
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图
        Returns:
            多尺度聚合后的特征图
        """
        # 获取输入特征图的宽度和高度
        width = x.shape[-1]
        height = x.shape[-2]
        
        # 存储各个尺度的特征图
        x_list = []

        # 尺度0：原始尺度
        x_list.append(self.scale0(x))
        
        # 尺度1：上采样后与尺度0相加，然后处理
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear') + x_list[0])))
        
        # 尺度2：上采样后与尺度1相加，然后处理
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear') + x_list[1]))))
        
        # 尺度3：上采样后与尺度2相加，然后处理
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear') + x_list[2]))))
        
        # 尺度4：上采样后与尺度3相加，然后处理
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear') + x_list[3]))))
       
        # 将所有尺度的特征图拼接，压缩，并加上快捷连接
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):
    """
    分割头部模块，用于生成最终的分割预测
    """
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        """
        初始化分割头部
        Args:
            inplanes: 输入通道数
            interplanes: 中间层通道数
            outplanes: 输出通道数（类别数）
            scale_factor: 上采样倍数，如果为None则不进行上采样
        """
        super(segmenthead, self).__init__()
        # 第一个批归一化层
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        # 第一个3x3卷积层
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        # 第二个批归一化层
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 最终的1x1卷积层，输出类别预测
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        # 上采样倍数
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图
        Returns:
            分割预测结果
        """
        # 批归一化 -> ReLU -> 3x3卷积
        x = self.conv1(self.relu(self.bn1(x)))
        # 批归一化 -> ReLU -> 1x1卷积
        out = self.conv2(self.relu(self.bn2(x)))

        # 如果指定了上采样倍数，进行双线性插值上采样
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out


class DualResNet(nn.Module):
    """
    双分辨率网络（DualResNet）主体结构
    实现了高分辨率和低分辨率两个分支的并行处理
    """
    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True):
        """
        初始化DualResNet网络
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            layers: 各层的残差块数量列表
            num_classes: 分割类别数，默认19（Cityscapes数据集）
            planes: 基础通道数，默认64
            spp_planes: SPP模块通道数，默认128
            head_planes: 分割头通道数，默认128
            augment: 是否使用辅助分割头，默认True
        """
        super(DualResNet, self).__init__()

        # 高分辨率分支的通道数（基础通道数的2倍）
        highres_planes = planes * 2
        # 是否使用辅助分割头
        self.augment = augment

        # 初始卷积层：两个3x3卷积，每个都进行2倍下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),  # 第一个卷积
            BatchNorm2d(planes, momentum=bn_mom),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),  # 第二个卷积
            BatchNorm2d(planes, momentum=bn_mom),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活
        )

        # ReLU激活函数（非原地操作）
        self.relu = nn.ReLU(inplace=False)
        
        # 低分辨率分支的各层
        self.layer1 = self._make_layer(block, planes, planes, layers[0])  # 第1层
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)  # 第2层
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)  # 第3层
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)  # 第4层

        # 压缩模块：将低分辨率分支的特征压缩到高分辨率分支的通道数
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),  # 1x1卷积
            BatchNorm2d(highres_planes, momentum=bn_mom),  # 批归一化
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),  # 1x1卷积
            BatchNorm2d(highres_planes, momentum=bn_mom),  # 批归一化
        )

        # 下采样模块：将高分辨率分支的特征下采样到低分辨率分支
        self.down3 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=bn_mom),
        )

        # 高分辨率分支的各层
        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)  # 高分辨率第3层
        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)  # 高分辨率第4层
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)  # 高分辨率第5层

        # 低分辨率分支的第5层
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        # 空间金字塔池化模块
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # 辅助分割头（如果启用）
        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        # 最终分割头
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                # 批归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """
        构建残差层
        Args:
            block: 残差块类型
            inplanes: 输入通道数
            planes: 输出通道数
            blocks: 残差块数量
            stride: 步长
        Returns:
            nn.Sequential: 残差层
        """
        downsample = None
        # 如果步长不为1或输入输出通道数不匹配，需要下采样
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        # 第一个残差块（可能包含下采样）
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        
        # 后续残差块
        for i in range(1, blocks):
            if i == (blocks - 1):
                # 最后一个残差块不使用ReLU
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                # 中间的残差块使用ReLU
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入图像，形状为[B, 3, H, W]
        Returns:
            分割预测结果
        """
        # 计算输出特征图的尺寸（输入的1/8）
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []  # 存储各层的输出

        # 初始卷积处理
        x = self.conv1(x)

        # 低分辨率分支的前向传播
        x = self.layer1(x)  # 第1层
        layers.append(x)

        x = self.layer2(self.relu(x))  # 第2层
        layers.append(x)

        x = self.layer3(self.relu(x))  # 第3层
        layers.append(x)
        
        # 高分辨率分支开始处理
        x_ = self.layer3_(self.relu(layers[1]))  # 从第2层开始高分辨率处理

        # 双分支信息交换：低分辨率 -> 高分辨率，高分辨率 -> 低分辨率
        x = x + self.down3(self.relu(x_))  # 高分辨率信息融入低分辨率
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),  # 低分辨率信息融入高分辨率
                        size=[height_output, width_output],
                        mode='bilinear')
        
        # 如果启用辅助分割头，保存当前高分辨率特征
        if self.augment:
            temp = x_

        # 继续低分辨率分支处理
        x = self.layer4(self.relu(x))  # 第4层
        layers.append(x)
        
        # 继续高分辨率分支处理
        x_ = self.layer4_(self.relu(x_))  # 高分辨率第4层

        # 再次进行双分支信息交换
        x = x + self.down4(self.relu(x_))  # 高分辨率信息融入低分辨率
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),  # 低分辨率信息融入高分辨率
                        size=[height_output, width_output],
                        mode='bilinear')

        # 高分辨率分支的最后一层
        x_ = self.layer5_(self.relu(x_))
        
        # 低分辨率分支经过第5层和SPP模块，然后上采样
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        # 融合两个分支的特征并生成最终预测
        x_ = self.final_layer(x + x_)

        # 如果启用辅助分割头，返回两个预测结果
        if self.augment:
            x_extra = self.seghead_extra(temp)  # 辅助预测
            return [x_extra, x_]  # 返回[辅助预测, 主预测]
        else:
            return x_  # 只返回主预测


def DualResNet_imagenet(cfg, pretrained=False):
    """
    构建在ImageNet上预训练的DualResNet模型
    Args:
        cfg: 配置对象，包含模型参数
        pretrained: 是否加载预训练权重
    Returns:
        DualResNet模型实例
    """
    # 创建DualResNet模型
    # BasicBlock: 使用基础残差块
    # [2, 2, 2, 2]: 各层的残差块数量
    # num_classes=19: Cityscapes数据集的类别数
    # planes=32: 基础通道数（轻量级版本）
    # spp_planes=128: SPP模块通道数
    # head_planes=64: 分割头通道数（轻量级版本）
    # augment=True: 启用辅助分割头
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, 
                       spp_planes=128, head_planes=64, augment=True)
    
    # 如果需要加载预训练权重
    if pretrained:
        # 加载预训练状态字典
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        # 获取当前模型的状态字典
        model_dict = model.state_dict()
        # 过滤预训练权重：只保留形状匹配的权重
        pretrained_state = {k: v for k, v in pretrained_state.items() 
                           if (k in model_dict and v.shape == model_dict[k].shape)}
        # 更新模型权重
        model_dict.update(pretrained_state)
        # 加载权重到模型（允许部分加载）
        model.load_state_dict(model_dict, strict=False)
    
    return model


def get_seg_model(cfg, **kwargs):
    """
    获取分割模型的工厂函数
    Args:
        cfg: 配置对象
        **kwargs: 其他关键字参数
    Returns:
        配置好的分割模型
    """
    # 创建并返回预训练的DualResNet模型
    model = DualResNet_imagenet(cfg, pretrained=True)
    return model


# 测试代码
if __name__ == '__main__':
    # 创建随机输入张量：批次大小4，3通道，800x800像素
    x = torch.rand(4, 3, 800, 800)
    # 创建预训练网络实例
    net = DualResNet_imagenet(pretrained=True)
    # 前向传播
    y = net(x)
    # 打印输出形状
    print(y.shape)