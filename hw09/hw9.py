import sys
import torch
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from preprocess import Image_Dataset,preprocess
from DR_Clustering import cal_acc,plot_scatter
from model import AE,AE_baseline
from utils import count_parameters,same_seeds
from test import inference,predict,invert,save_prediction

if __name__ == "__main__":
    if(len(sys.argv)<2):
        print("Usage: python3 hw9.py -train <trainX_npy> <checkpoint> -baseline/improved")
        print("Usage: python3 hw9.py -test <trainX_npy> <checkpoint> <prediction_path>")
        print("Usage: python3 hw9.py -report <trainX_npy> <checkpoint>")
        exit()

    trainX = np.load(sys.argv[2])
    trainX_preprocessed = preprocess(trainX)
    
   
    img_dataset = Image_Dataset(trainX_preprocessed)

    if(sys.argv[1]=="-train"):
        
        same_seeds(0)
        if(sys.argv[4]=="-baseline"):
            print("Constructing the baseline model...")
            model = AE_baseline().cuda()
        elif(sys.argv[4]=="-improved"):
            print("Constructing the improved model...")
            model = AE().cuda()
        else:
            print("Wrong motion!!")
            exit()
        #print(model)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        n_epoch = 100
        print("Preparing dataloader...")
        # 準備 dataloader, model, loss criterion 和 optimizer
        img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

        print("Training...")
        # 主要的訓練過程
        for epoch in range(n_epoch):
            for data in img_dataloader:
                img = data
                img = img.cuda()
                #imgn = img + torch.autograd.Variable(torch.randn(img.size())).cuda()
                #imgn = img + torch.randn(img.size()).cuda()
                output1, output = model(img)
                loss = criterion(output, img)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch+1) % 10 == 0:
                    torch.save(model.state_dict(), './checkpoints_save/checkpoint_{}.pth'.format(epoch+1))
                    
            print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

        # 訓練完成後儲存 model
        torch.save(model.state_dict(), sys.argv[3])
    
    elif(sys.argv[1]=="-test"):
        csv_path = sys.argv[4]
        csv_inv_path = csv_path[:len(csv_path)-4]+'_invert.csv'
        
        # load model
        model = AE().cuda()
        model.load_state_dict(torch.load(sys.argv[3]))
        model.eval()

        # 準備 data
        trainX = np.load(sys.argv[2])
        #print(trainX.shape)
        # 預測答案
        latents = inference(X=trainX, model=model)
        pred, X_embedded = predict(latents)

        # 將預測結果存檔，上傳 kaggle
        save_prediction(pred, csv_path)

        # 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
        # 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
        save_prediction(invert(pred), csv_inv_path)