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
import pandas as pd
from torchvision.models.mobilenetv2 import *
from torchvision.models.mobilenetv3 import *
#from torchvision.models.mnasnet import *
from mnasnet import *

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
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
    today = 22062503
    model = MnasNet(n_class=10, input_size=32, width_mult=0.01).to(device)
    print(model)
    # LOAD PRE_TRAIN
    #model.load_state_dict(torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\epochs\\22060605\epoch_22060605_100.pth'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    results = {'loss': [], 'accuracy': []}
    # pre-train the model
    pre_train_epoch = 200
    if pre_train_epoch:
        os.mkdir('./epochs/' + str(today))
        for _ in range(pre_train_epoch):
            loss = trainer(model, optimizer, criterion)
            results['loss'].append(loss)
            accuracy = evaluator(model)
            results['accuracy'].append(accuracy)
            torch.save(model.state_dict(), 'epochs/' + str(today) + '/epoch_%d_%d.pth' % (today, epoch))
        data_frame = pd.DataFrame(data={'Loss': results['loss'], 'Accuracy': results['accuracy']}, index=range(1, pre_train_epoch + 1))
        data_frame.to_csv('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\\results\\results_%d.csv' % today, index_label='Epoch')