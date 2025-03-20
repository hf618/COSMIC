# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from .dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from .dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg
# from dinov2.dinov2.hub.classifiers import dinov2_vitb14_lc, dinov2_vitg14_lc, dinov2_vitl14_lc, dinov2_vits14_lc
# from dinov2.dinov2.hub.classifiers import dinov2_vitb14_reg_lc, dinov2_vitg14_reg_lc, dinov2_vitl14_reg_lc, dinov2_vits14_reg_lc
# from dinov2.dinov2.hub.depthers import dinov2_vitb14_ld, dinov2_vitg14_ld, dinov2_vitl14_ld, dinov2_vits14_ld
# from dinov2.dinov2.hub.depthers import dinov2_vitb14_dd, dinov2_vitg14_dd, dinov2_vitl14_dd, dinov2_vits14_dd


dependencies = ["torch"]


import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

def load_dinov2(size):
    #model = dinov2_vitb14_reg(weights={'LVD142M': './pretrained/dinov2_vitb14_reg4_pretrain.pth'})
    if size == "l":
        model = dinov2_vitl14_reg(weights={'LVD142M': './pretrained/dinov2_vitl4_reg4_pretrain.pth'})
    elif size == "b":
        model = dinov2_vitb14_reg(weights={'LVD142M': './pretrained/dinov2_vitb14_reg4_pretrain.pth'})
    elif size == "s":
        model = dinov2_vits14_reg(weights={'LVD142M': './pretrained/dinov2_vits14_reg4_pretrain.pth'})
    # model = dinov2_vits14_reg(weights={'LVD142M': './pretrained/dinov2_vits14_reg4_pretrain.pth'})
    # model = dinov2_vitg14_reg(weights={'LVD142M': './pretrained/dinov2_vitg14_reg4_pretrain.pth'})

    return model