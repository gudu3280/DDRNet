# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import seg_hrnet
from . import seg_hrnet_ocr
from . import ddrnet_23_slim
from . import ddrnet_23
from . import ddrnet_39

def get_seg_model(cfg, **kwargs):
    """根据配置获取分割模型"""
    model_name = cfg.MODEL.NAME
    
    if model_name == 'seg_hrnet':
        return seg_hrnet.get_seg_model(cfg, **kwargs)
    elif model_name == 'seg_hrnet_ocr':
        return seg_hrnet_ocr.get_seg_model(cfg, **kwargs)
    elif model_name == 'ddrnet_23_slim':
        return ddrnet_23_slim.get_seg_model(cfg, **kwargs)
    elif model_name == 'ddrnet_23':
        return ddrnet_23.get_seg_model(cfg, **kwargs)
    elif model_name == 'ddrnet_39':
        return ddrnet_39.get_seg_model(cfg, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")