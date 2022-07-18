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
#from torchvision.models.mnasnet import *
from mnasnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR = 'C'
SIZE = 32
PATH = "D:\HUST\CH-Motor_Research\Testbench data\Data_augument\IMF1"
test_set = TrainDataset(PATH, COLOR, SIZE, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

def count_macs(model, input_sz, mask=None):
    input=torch.randn(1, 3, input_sz, input_sz)
    _, _, rst = count_flops_params(model, input)
    total_param = 0
    total_macs = 0
    layer_num = -1
    for layer in mask:
        layer_num += 1
        #print(layer)
        #print(rst[layer_num]['name'])
        dim = mask[layer]['weight'].dim()
        param = mask[layer]['weight']
        output_sz = rst[layer_num]['output_size']
        if 4==dim:  #Conv2d
            for num in param:
                for c in num:
                    for h in c:
                        for w in h:
                            if w == 1:
                                total_param += 1
                                total_macs += output_sz[-1]**2
        if 2==dim:  #Linear
            for h in param:
                for w in h:
                    if w == 1:
                        total_param += 1
                        total_macs += 1
    print("PARAM:", total_param)
    print("MACs:", total_macs)

def evaluator(model):
    print('[LOG]: evaluator')
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(iterable=test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print('Accuracy: {}%\n'.format(acc))
    return acc

mask = dict()
mask = torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\\2022-06-11-15-10-59-567907\\best_result\masks.pth')
model = mobilenet_v3_small(n_class=10, width_mult=0.05)
#model.load_state_dict(torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\epochs\\22062305\epoch_22062305_1.pth'))
count_macs(model,32, mask)
#evaluator(model)
#input=torch.randn(1, 3, SIZE, SIZE)
#count_flops_params(model, input)
