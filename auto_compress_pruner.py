import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from dataset import TrainDataset
from torch.optim.lr_scheduler import MultiStepLR
import os
import nni
from nni.compression.pytorch.pruning import AutoCompressPruner, FPGMPruner
from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch import ModelSpeedup
from torchvision.models.mobilenetv2 import *
from torchvision.models.mobilenetv3 import *
from lenet import LeNet
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR = 'G'
SIZE = 32
PATH = "D:\HUST\CH-Motor_Research\Testbench data\Data_augument\IMF1"

train_set = TrainDataset(PATH, COLOR, SIZE, 'train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_set = TrainDataset(PATH, COLOR, SIZE, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()

epoch = 0

def trainer(model, optimizer, criterion):
    print('[LOG]: trainer')
    global epoch
    model.train()
    total_loss = 0
    total_size = 0
    for data, target in tqdm(iterable=train_loader, desc='Total Epoch {}'.format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()*data.size(0)
        loss.backward()
        optimizer.step()
        total_size += data.size(0)
    epoch = epoch + 1
    return total_loss/total_size

def finetuner(model):
    print('[LOG]: finetuner')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(10):
        trainer(model, optimizer, criterion)

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

if __name__ == '__main__':
    model = mobilenet_v3_small(n_class=10, width_mult=0.05).to(device)

    model.load_state_dict(torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\epochs\\22061012\epoch_22061012_78.pth'))  
    config_list = [{'op_types': ['Linear', 'Conv2d'], 'total_sparsity': 0.1}]

    # make sure you have used nni.trace to wrap the optimizer class before initialize
    traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    admm_params = {
        'trainer': trainer,
        'traced_optimizer': traced_optimizer,
        'criterion': criterion,
        'iterations': 5,
        'training_epochs': 5
    }
    sa_params = {
        'evaluator': evaluator
    }
    pruner = AutoCompressPruner(model, config_list, 1, admm_params, sa_params, keep_intermediate_result=True)
    print('[LOG]: compress')
    pruner.compress()
    _, model, masks, _, _ = pruner.get_best_result()
    evaluator(model)

