# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""
路径初始化脚本
用于将项目的lib目录添加到Python模块搜索路径中
确保可以正确导入项目的自定义模块
"""

from __future__ import absolute_import  # 绝对导入
from __future__ import division         # 除法运算
from __future__ import print_function   # print函数

import os.path as osp  # 路径操作模块
import sys             # 系统相关模块


def add_path(path):
    """
    将指定路径添加到Python模块搜索路径中
    
    参数:
        path (str): 要添加的路径
    
    功能:
        如果路径不在sys.path中，则将其插入到列表开头
        这样可以确保优先从该路径导入模块
    """
    if path not in sys.path:
        sys.path.insert(0, path)  # 插入到搜索路径的开头

# 获取当前脚本所在目录
this_dir = osp.dirname(__file__)

# 构建lib目录的路径（相对于当前目录的上级目录）
lib_path = osp.join(this_dir, '..', 'lib')

# 将lib目录添加到Python模块搜索路径中
# 这样就可以导入lib目录下的models、datasets、config等模块
add_path(lib_path)
