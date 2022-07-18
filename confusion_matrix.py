import numpy as np
import matplotlib.pyplot as plt
from dataset import TrainDataset
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torchvision.models.mobilenetv2 import *
from torchvision.models.mobilenetv3 import *
from torchvision.models.mnasnet import *
from tqdm import tqdm

COLOR = 'G'
SIZE = 32
PATH = "D:\HUST\CH-Motor_Research\Testbench data\Data_augument\IMF1"
test_set = TrainDataset(PATH, COLOR, SIZE, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mnasnet1_0(num_classes=10).to(device)
model.load_state_dict(torch.load('D:\HUST\CH-Motor_Research\Code\IBFD_MobileNetV3_AutoCompress\epochs\\22062301\epoch_22062301_6.pth'))
model.eval()

Y_true = []
Y_pred = []

with torch.no_grad():
    for data, target in tqdm(iterable=test_loader, desc='Test'):
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        Y_true.append(target.item())
        Y_pred.append(pred.item())
    

class_names = ['NO', 'I7', 'B7', 'O7', 'I14', 'B14', 'O14', 'I21', 'B21', 'O21'] # 0: NO | 1: IR7 | 2: B7 | 3: OR7 | 4: IR14 | 5: B14 | 6: OR14 | 7: IR21 | 8: B21 | 9: OR21


# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_predictions(
        Y_true,
        Y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.show()
#plt.savefig('test.png', dpi=300)


# calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn
#from sklearn.metrics import recall_score, f1_score

#recall = recall_score(Y_true, Y_pred, labels=[0,1,2,3,4,5,6,7,8,9], average='micro')
#print('Recall: %f' % recall)
#score = f1_score(Y_true, Y_pred,labels=[0,1,2,3,4,5,6,7,8,9], average='micro')
#print('F-Measure: %f' % score)