import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset,TensorDataset
import time

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

class Classifier_ResNet(nn.Module):
    def __init__(self):
        super(Classifier_ResNet, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.conv364 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv3264 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv64128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv64128_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv128256 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv128256_1 = nn.Conv2d(128, 256, 3, 1, 1)
       
        self.conv6432 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv12864 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv256128 = nn.Conv2d(256, 128, 3, 1, 1)
    
        self.BN32_0 = nn.BatchNorm2d(32)
        self.BN32_1 = nn.BatchNorm2d(32)
        self.BN32_2 = nn.BatchNorm2d(32)
        self.BN64_0 = nn.BatchNorm2d(64)
        self.BN64_1 = nn.BatchNorm2d(64)
        self.BN64_2 = nn.BatchNorm2d(64)
        self.BN128_0 = nn.BatchNorm2d(128)
        self.BN128_1 = nn.BatchNorm2d(128)
        self.BN128_2 = nn.BatchNorm2d(128)
        self.BN128_3 = nn.BatchNorm2d(128)
        self.BN256_0 = nn.BatchNorm2d(256)
        self.BN256_1 = nn.BatchNorm2d(256)
        self.BN256_2 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.MP2D = nn.MaxPool2d(2,2,0)

        self.fc = nn.Sequential(
            nn.Linear(256*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        
        x = self.conv364(x)     # [64, 128, 128]
        x = self.BN64_0(x)
        x_res64 = self.relu(x) 
        
        x = self.conv6432(x_res64)
        x = self.BN32_0(x)
        x = self.relu(x)

        x = self.conv3264(x)
        x = self.BN64_1(x)
        x = self.relu(x)

        x = x+x_res64
        #x = self.dropout(x)
        x = self.MP2D(x)        # [64, 64, 64]

        x = self.conv64128(x)     # [128, 64, 64]
        x = self.BN128_0(x)
        x_res128 = self.relu(x) 
        
        x = self.conv12864(x_res128)
        x = self.BN64_2(x)
        x = self.relu(x)

        x = self.conv64128_1(x)
        x = self.BN128_1(x)
        x = self.relu(x)

        x = x+x_res128
        #x = self.dropout(x)
        x = self.MP2D(x)        # [128, 32, 32]

        x = self.conv128256(x)     # [256, 32, 32]
        x = self.BN256_0(x)
        x_res256 = self.relu(x) 
        
        x = self.conv256128(x_res256)
        x = self.BN128_2(x)
        x = self.relu(x)

        x = self.conv128256_1(x)
        x = self.BN256_1(x)
        x = self.relu(x)

        x = x+x_res256
        #x = self.dropout(x)
        x_res256_1 = self.MP2D(x)        # [256, 16, 16]
        
        x = self.conv256128(x_res256_1)
        x = self.BN128_3(x)
        x = self.relu(x)

        x = self.conv128256_1(x)
        x = self.BN256_2(x)
        x = self.relu(x)

        x = x+x_res256_1
        #x = self.dropout(x)
        out = self.MP2D(x)        # [256, 8,8]
        
        out = out.view(out.size()[0], -1)
        out = self.dropout(out)
        return self.fc(out)

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

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(66,11),
            nn.Softmax()
        )
        self.dnn[0].weight.data = torch.Tensor(np.concatenate([np.identity(11)]*6,axis=1))
        
    def forward(self, x):
        return self.dnn(x)



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

def train(train_set,val_set,train_loader,val_loader,lr,weight_decay,num_epoch,model_path):
    model = Classifier().cuda()
    total = sum(p.numel() for p in model.parameters())
    print("Size of parameters = ",total)
    
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-4) # optimizer 使用 Adam
    num_epoch = 50

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
            optimizer.step() # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

                #將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
    torch.save(model.state_dict(), model_path)
    print("Already save model to ",model_path)
        
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)

  
    train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

    model_best = Classifier().cuda()
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model_best.parameters(), lr=lr, weight_decay=weight_decay) # optimizer 使用 Adam

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0

        model_best.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model_best(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
        (epoch + 1, num_epoch, time.time()-epoch_start_time, \
        train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))


    torch.save(model_best.state_dict(), model_path)
    print("")
    print("Already save model to ",model_path)

def test(test_set,test_loader,batch_size,argmax,test_model_path,VGG):    
    if(VGG):
        model_best = Classifier().cuda()
    else:
        model_best = Classifier_ResNet().cuda()
    model_best.load_state_dict(torch.load(test_model_path)) 

    model_best.eval()
    prediction = []
    prediction_all = []
    print("Start testing")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data.cuda())
            test_all = test_pred.cpu().data.numpy()
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_all:
                prediction_all.append(y)
            for y in test_label:
                prediction.append(y)
    if(argmax):
        return prediction
    else:
        return prediction_all

def train_ensemble(train_set,val_set,train_loader,val_loader,lr,weight_decay,num_epoch,model_path):

    model = Ensemble().cuda()
    total = sum(p.numel() for p in model.parameters())
    print("Size of parameters = ",total)
    
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-4) # optimizer 使用 Adam

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
            optimizer.step() # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

                #將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
    torch.save(model.state_dict(), model_path)
    print("Already save model to ",model_path)

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
        output_path = sys.argv[4]
    if(sys.argv[1]=="-train"):
        print("Train")
    if(sys.argv[1]=="-ensemble_test"):
        print("Ensemble test Only")
        output_path = sys.argv[4]
    if(sys.argv[1]=="-ensemble_test1"):
        print("Ensemble test Only")
        output_path = sys.argv[4]
    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))

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

    if(sys.argv[1]=="-train"):
        print("Start training")
        batch_size = 64
        train_set = ImgDataset(train_x, train_y, train_transform)
        val_set = ImgDataset(val_x, val_y, test_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        lr=0.0005
        weight_decay=1e-4
        num_epoch = 90
        train(train_set,val_set,train_loader,val_loader,lr,weight_decay,num_epoch,model_path)


    if(sys.argv[1]=="-test"):
        print("Start loading")
        batch_size = 64
        test_set = ImgDataset(test_x, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        prediction = test(test_set,test_loader,batch_size,True,model_path,True)

        #將結果寫入 csv 檔
        with open(output_path, 'w') as f:
            f.write('Id,Category\n')
            for i, y in  enumerate(prediction):
                f.write('{},{}\n'.format(i, y))

    if(sys.argv[1]=="-ensemble_train"):
        
        print("Ensembling")
        batch_size = 64
        
        train_set = ImgDataset(train_x,transform=test_transform)
        val_set = ImgDataset(val_x,transform=test_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        print(len(train_set))
        prediction_train = np.asarray(test(train_set,train_loader,batch_size,False,"model/model_VGG.pkl",True))
        prediction_val = np.asarray(test(val_set,val_loader,batch_size,False,"model/model_VGG.pkl",True))
        prediction_train1 = np.asarray(test(train_set,train_loader,batch_size,False,"model/model_VGG_2.pkl",True))
        prediction_val1 = np.asarray(test(val_set,val_loader,batch_size,False,"model/model_VGG_2.pkl",True))
        prediction_train = np.concatenate((prediction_train, prediction_train1), axis=1)
        prediction_val = np.concatenate((prediction_val, prediction_val1), axis=1)
        prediction_train1 = np.asarray(test(train_set,train_loader,batch_size,False,"model/model1.pkl",False))
        prediction_val1 = np.asarray(test(val_set,val_loader,batch_size,False,"model/model1.pkl",False))
        prediction_train = np.concatenate((prediction_train, prediction_train1), axis=1)
        prediction_val = np.concatenate((prediction_val, prediction_val1), axis=1)
        prediction_train1 = np.asarray(test(train_set,train_loader,batch_size,False,"model/model2.pkl",False))
        prediction_val1 = np.asarray(test(val_set,val_loader,batch_size,False,"model/model2.pkl",False))
        prediction_train = np.concatenate((prediction_train, prediction_train1), axis=1)
        prediction_val = np.concatenate((prediction_val, prediction_val1), axis=1)
        prediction_train1 = np.asarray(test(train_set,train_loader,batch_size,False,"model/model3.pkl",False))
        prediction_val1 = np.asarray(test(val_set,val_loader,batch_size,False,"model/model3.pkl",False))
        prediction_train = np.concatenate((prediction_train, prediction_train1), axis=1)
        prediction_val = np.concatenate((prediction_val, prediction_val1), axis=1)
        prediction_train1 = np.asarray(test(train_set,train_loader,batch_size,False,"model/model4.pkl",False))
        prediction_val1 = np.asarray(test(val_set,val_loader,batch_size,False,"model/model4.pkl",False))
        prediction_train = np.concatenate((prediction_train, prediction_train1), axis=1)
        prediction_val = np.concatenate((prediction_val, prediction_val1), axis=1)
        print(len(train_set))
        
        
        train_set = TensorDataset(torch.from_numpy(prediction_train), (torch.from_numpy(train_y)).type(torch.LongTensor))
        val_set = TensorDataset(torch.from_numpy(prediction_val), (torch.from_numpy(val_y)).type(torch.LongTensor))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        lr=0.0005
        weight_decay=1e-4
        num_epoch = 10
        train_ensemble(train_set,val_set,train_loader,val_loader,lr,weight_decay,num_epoch,model_path)
    
    if(sys.argv[1]=="-ensemble_test"):
        print("Ensembling")
        batch_size = 64
        test_set = ImgDataset(test_x, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        prediction_test = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model_VGG.pkl",True))
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model_VGG_2.pkl",True))
        prediction_test = np.concatenate((prediction_test, prediction_test1), axis=1)
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model1.pkl",False))
        prediction_test = np.concatenate((prediction_test, prediction_test1), axis=1)
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model2.pkl",False))
        prediction_test = np.concatenate((prediction_test, prediction_test1), axis=1)
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model3.pkl",False))
        prediction_test = np.concatenate((prediction_test, prediction_test1), axis=1)
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"./model/model4.pkl",False))
        prediction_test = np.concatenate((prediction_test, prediction_test1), axis=1)
        
        test_loader = DataLoader(prediction_test, batch_size=batch_size, shuffle=False)
        model_best = Ensemble().cuda()
        model_best.load_state_dict(torch.load("./model/model_ensemble.pkl")) 

        model_best.eval()
        prediction = []
        print("Start testing")
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_pred = model_best(data.cuda())
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                for y in test_label:
                    prediction.append(y)
        
        
        
        with open(output_path, 'w') as f:
            f.write('Id,Category\n')
            for i, y in  enumerate(prediction):
                f.write('{},{}\n'.format(i, y))

    if(sys.argv[1]=="-ensemble_test1"):
        print("Ensembling")
        batch_size = 64
        test_set = ImgDataset(test_x, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        prediction_test = np.asarray(test(test_set,test_loader,batch_size,False,"model_VGG.pkl",True))
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"model_VGG_2.pkl",True))
        prediction_test = prediction_test + prediction_test1
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"model1.pkl",False))
        prediction_test = prediction_test + prediction_test1
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"model2.pkl",False))
        prediction_test = prediction_test + prediction_test1
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"model3.pkl",False))
        prediction_test = prediction_test + prediction_test1
        prediction_test1 = np.asarray(test(test_set,test_loader,batch_size,False,"model4.pkl",False))
        prediction_test = prediction_test + prediction_test1
       
        prediction = []
        for i in range(len(prediction_test)):
            test_label = np.argmax(prediction_test[i])
            prediction.append(test_label)

        with open(output_path, 'w') as f:
            f.write('Id,Category\n')
            for i, y in  enumerate(prediction):
                f.write('{},{}\n'.format(i, y))
    