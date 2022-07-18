from typing import Dict
import numpy as np
from pyrsistent import m
import torch
import time
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.pruning import AutoCompressPruner, AMCPruner
from nni.compression.pytorch.utils import count_flops_params
from thop import profile
from lenet import *
from dataset import TrainDataset
from tqdm import tqdm
from nni.compression.pytorch.utils import count_flops_params
from torchvision.models.mobilenetv2 import *
from torchvision.models.mobilenetv3 import *
from torchvision.models.mnasnet import *
import matplotlib.pyplot as plt

model = mobilenet_v3_small(n_class = 10, width_mult = 0.05)
model.load_state_dict(torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\epochs\\22061012\epoch_22061012_78.pth'))
t = np.linspace(-2, 2, 100)
n = np.linspace(-2, 2, 100)

for name, param in model.named_parameters():
    dat = param.data
    if 4 == dat.dim():
        for num in dat:
            for c in num:
                for h in c:
                    for w in h:
                        for ti in range(100):
                            if w>t[ti] and w<t[ti+1]:
                                n[ti] += 1 
    if 2 == dat.dim():  #Linear
        for h in dat:
            for w in h:
                for ti in range(100):
                    if w>t[ti] and w<t[ti+1]:
                        n[ti] += 1
plt.stem(t, n)
plt.show()