import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            #nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            #nn.Dropout2d(p=0.3),
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            #nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            #nn.Dropout2d(p=0.3),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 11)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out) 

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


if __name__ == '__main__':
    #python3 hw3.py -train ./food-11 model.pkl
    #python3 hw3.py -test  ./food-11 model.pkl predict.csv
    if(len(sys.argv)<4):
        print("Error")
        print("Usage: python3 hw3.py -train ./food-11 model.pkl")
        print("Usage: python3 hw3.py -test  ./food-11 model.pkl predict.csv")
        exit()
    workspace_dir = sys.argv[2]
    model_path = sys.argv[3]
    if(sys.argv[1]=="-test"):
        print("Test Only")
        output_prefix = sys.argv[4]
    if(sys.argv[1]=="-train"):
        print("Train")

    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    

    #training 時做 data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
        transforms.RandomRotation(15), #隨機旋轉圖片
        transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
        
    ])
    #testing 時不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
        
    ])

    
    if(sys.argv[1]=="-test"):
        print("Start loading")
        batch_size = 64
        test_set = ImgDataset(val_x, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        model_best = Classifier().cuda()
        model_best.load_state_dict(torch.load(model_path)) 

        model_best.eval()
        prediction = []
        print("Start testing")
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_pred = model_best(data.cuda())
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                for y in test_label:
                    prediction.append(y)

        #將結果寫入 csv 檔
    cnfm = confusion_matrix(val_y,prediction)

    cnfm = cnfm.astype('float')/cnfm.sum(axis=1)[:,np.newaxis]

    #print(cnfm)
    type_food = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
    plt.imshow(cnfm,interpolation='nearest',cmap='Blues')
    plt.title("Confusion Matrix")

    plt.colorbar()
    tick_marks = np.arange(len(type_food))
    plt.xticks(tick_marks,type_food,rotation=90)
    plt.yticks(tick_marks,type_food)
    ax = plt.gca()
    ax.set_yticklabels(type_food)
    ax.set_xticklabels(type_food)
    ax.set_ylim(len(type_food)-0.5, -0.5)
    fmt = '.2f'
    thresh = cnfm.max()/2.
    for i,j in product(range(cnfm.shape[0]),range(cnfm.shape[1])):
        plt.text(j,i,format(cnfm[i,j],fmt),ha='center',
        color="white" if cnfm[i,j] >thresh else "black", fontsize=7)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix,"cnfm.png"))
    plt.close()
    
    