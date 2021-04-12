# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


import torch
from torch import nn
import torchvision.models as models


def load_model(model_name, num_categories, use_pretrained=True):
    if model_name == 'resnet50-0':
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load('./pretrained/resnet50.pth'))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
    else:
        model = None
    return model

