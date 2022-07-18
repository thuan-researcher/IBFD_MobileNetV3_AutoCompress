from os import listdir
from os.path import join
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, data_path, color, size, mode):  # color: 'C', 'G'  |  size:  32, 64, 100,  mode: 'train', 'test'
        super(TrainDataset, self).__init__()
        self.filenames, self.lables = get_image(data_path, color, size, mode)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.lables[index]  
        image = ToTensor()(image)
        return image, label

    def __len__(self):
        return len(self.filenames)

def get_image(data_path, color, size, mode):
    data_list = []
    label_list = []
    datapath = listdir(join(data_path,  "{:s}{:d}".format(color, size), mode))
    for x in datapath:
        data_list.append(join(data_path, "{:s}{:d}".format(color, size), mode, x))
        code = int(x[1:4])     # 0: NO | 1: IR7 | 2: B7 | 3: OR7 | 4: IR14 | 5: B14 | 6: OR14 | 7: IR21 | 8: B21 | 9: OR21
        if code <= 100:
            label_list.append(0)
        elif code <= 112:
            label_list.append(1)
        elif code <= 125:
            label_list.append(2)
        elif code <= 138:
            label_list.append(3)
        elif code <= 177:
            label_list.append(4)
        elif code <= 192:
            label_list.append(5)
        elif code <= 204:
            label_list.append(6)
        elif code <= 217:
            label_list.append(7)
        elif code <= 229:
            label_list.append(8)
        elif code <= 241:
            label_list.append(9)

    return data_list, torch.LongTensor(label_list)