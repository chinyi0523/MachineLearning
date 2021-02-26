# main.py
import sys
from data import TwitterDataset
from test import testing,testing_ens
from train import training,training_ens
from model import LSTM_Net,Ensemble
from model import GRU_Net
from preprocess import Preprocess
from utils import load_training_data,load_testing_data,evaluation
import pandas as pd
import w2v
import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset,TensorDataset

if __name__ == "__main__":

    if(len(sys.argv) < 4):
        print("Usage: python3 main.py -train <training label data>  <training unlabel data> 30 try.model")
        print("Usage: python3 main.py -test <testing data>  <prediction file>" )
        print("Usage: python3 main.py -ensemble_train <training label data>  <training unlabel data> <testing data>")
        print("Usage: python3 main.py -ensemble_test <testing data>  <prediction file>")
        exit()
    path_prefix = './'
    if(sys.argv[1]=="-train"):
        train_with_label = os.path.join(path_prefix, sys.argv[2])
        train_no_label = os.path.join(path_prefix, sys.argv[3])
    elif(sys.argv[1]=="-test"):
        testing_data = os.path.join(path_prefix, sys.argv[2])
        output_file = sys.argv[3]
    elif(sys.argv[1]=="-ensemble_train"):
        train_with_label = os.path.join(path_prefix, sys.argv[2])
        train_no_label = os.path.join(path_prefix, sys.argv[3])
        testing_data = os.path.join(path_prefix, sys.argv[4])
    elif(sys.argv[1]=="-ensemble_test"):
        testing_data = os.path.join(path_prefix, sys.argv[2])
        output_file = sys.argv[3]
    else:
        exit()

    

    # 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個 data 的路徑
    w2v_path = []
    w2v_path.append(os.path.join(path_prefix, 'w2v_all.model'))
    w2v_path.append(os.path.join(path_prefix, 'w2v_cbow.model')) # 處理 word to vec model 的路徑
    
    # 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
    sen_len = 35
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 10
    lr = 0.001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = path_prefix # model directory for checkpoint model

    if(sys.argv[1]=="-train"):
        sen_len_input = int(sys.argv[4])
        model_name = sys.argv[5]
        sen_len = sen_len_input
        print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
        train_x, y = load_training_data(train_with_label)
        train_x_no_label = load_training_data(train_no_label)
        print("Preprocessing ...")
        # 對 input 跟 labels 做預處理
        preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path[0])
        embedding = preprocess.make_embedding(load=True)
        train_x = preprocess.sentence_word2idx()
        y = preprocess.labels_to_tensor(y)

        # 製作一個 model 的對象
        model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
        model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

        # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
        X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

        # 把 data 做成 dataset 供 dataloader 取用
        train_dataset = TwitterDataset(X=X_train, y=y_train)
        val_dataset = TwitterDataset(X=X_val, y=y_val)

        # 把 data 轉成 batch of tensors
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 8)

        val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False,
                                                    num_workers = 8)

        # 開始訓練
        training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device,model_name)

    if(sys.argv[1]=="-test"):
    # 開始測試模型並做預測
        model_dir = path_prefix
        print("loading testing data ...")
        test_x = load_testing_data(testing_data)
        preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path[0])
        embedding = preprocess.make_embedding(load=True)
        test_x = preprocess.sentence_word2idx()
        test_dataset = TwitterDataset(X=test_x, y=None)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False,
                                                    num_workers = 8)
        print('\nload model ...')
        model = torch.load(os.path.join(model_dir, 'ckpt.model'))
        outputs = testing(batch_size, test_loader, model, device, False)

        # 寫到 csv 檔案供上傳 Kaggle
        tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
        print("save csv ...")
        tmp.to_csv(os.path.join(path_prefix, output_file), index=False)
        print("Finish Predicting")
    
    if(sys.argv[1]=="-ensemble_train"):
        output_train=[]
        output_val=[]
        
        for i in range(len(w2v_path)):
            print("w2v_path = ",w2v_path[i])
            print("loading training data ...")
            train_x, y = load_training_data(train_with_label)
            train_x_no_label = load_training_data(train_no_label)
            print("Preprocessing ...")
            # 對 input 跟 labels 做預處理
            preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path[i])
            embedding = preprocess.make_embedding(load=True)
            train_x = preprocess.sentence_word2idx()
            y = preprocess.labels_to_tensor(y)
        
            # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
            X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]
            
            # 把 data 做成 dataset 供 dataloader 取用
            val_dataset = TwitterDataset(X=X_val, y=None)
            train_dataset = TwitterDataset(X=X_train, y=None)
            train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = False,num_workers = 8)                                           
            val_loader = torch.utils.data.DataLoader(dataset = val_dataset,batch_size = batch_size,shuffle = False,num_workers = 8) 
                                                                                               
            model_dir = path_prefix
        
            print('\nload model ...')##'ckpt30.model''ckpt5.model''ckpt10.model''ckptlen30.model''ckptlen35.model'
            
            if(i==0):
                
                model_name_list = ['ckpt_pre_20_0.model','ckpt_pre_25_0.model','ckpt_pre_25_1.model','ckpt_pre_25_2.model',
                'ckpt_pre_25_3.model','ckpt_pre_30_0.model','ckpt_pre_30_1.model','ckpt_pre_30_2.model','ckpt_pre_30_3.model',
                'ckpt_pre_35_0.model','ckpt_pre_35_1.model','ckpt_pre_35_2.model','ckpt_pre_35_3.model']
                
            if(i==1):
                model_name_list = ['ckpt_pre_20_cbow_0.model','ckpt_pre_25_cbow_0.model','ckpt_pre_25_cbow_1.model',
                'ckpt_pre_30_cbow_0.model','ckpt_pre_30_cbow_1.model','ckpt_pre_35_cbow_0.model','ckpt_pre_35_cbow_1.model']
            for i in range(len(model_name_list)):
                model = torch.load(os.path.join(model_dir, model_name_list[i]))
                outputs_tr = testing(batch_size, train_loader, model, device, True)
                outputs_tr = np.reshape(outputs_tr,(-1,len(outputs_tr))).T
                outputs_va = testing(batch_size, val_loader, model, device, True)
                outputs_va = np.reshape(outputs_va,(-1,len(outputs_va))).T
                output_train.append(outputs_tr)
                output_val.append(outputs_va)

        outputs = np.concatenate(output_train[:],axis=1)
        outputsv = np.concatenate(output_val[:],axis=1)
        outputs = outputs -0.5
        outputsv = outputsv -0.5
        
        ##outputs = np.asarray(outputs)+np.asarray(outputs1)+np.asarray(outputs2)+np.asarray(outputs3)+np.asarray(outputs4)+np.asarray(outputs5)+np.asarray(outputs6)+np.asarray(outputs7)+np.asarray(outputs8)+np.asarray(outputs9)+np.asarray(outputs10)+np.asarray(outputs11)+np.asarray(outputs12)
        print(outputs.shape)
        print(outputs[:10])
        print(outputsv.shape)
        print(outputsv[:10])
        print("Dealing with sets...")
        
        train_set = TensorDataset(torch.from_numpy(outputs).float(), y_train.float())
        val_set = TensorDataset(torch.from_numpy(outputsv).float(), y_val.float())
        train_loader = torch.utils.data.DataLoader(dataset = train_set,batch_size = batch_size,shuffle = True,num_workers = 8)
        val_loader = torch.utils.data.DataLoader(dataset = val_set,batch_size = batch_size,shuffle = False,num_workers = 8)
        
        print("Training...")
        model = Ensemble()
        model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

        batch_size = 128
        epoch = 5
        lr = 0.0005
        weight_decay=1e-4
        training_ens(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device,"ckpt_ensemble.model")


    if(sys.argv[1]=="-ensemble_test"):
        output_test=[]
        for i in range(len(w2v_path)):
            print("w2v_path = ",w2v_path[i])
            model_dir = path_prefix
            print("loading testing data ...")
            test_x = load_testing_data(testing_data)
            preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path[i])
            embedding = preprocess.make_embedding(load=True)
            test_x = preprocess.sentence_word2idx()
            test_dataset = TwitterDataset(X=test_x, y=None)
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False,num_workers = 8)
            print('\nload model ...')##'ckpt30.model''ckpt5.model''ckpt10.model''ckptlen30.model''ckptlen35.model'
        
            if(i==0):
                
                model_name_list = ['ckpt_pre_20_0.model','ckpt_pre_25_0.model','ckpt_pre_25_1.model','ckpt_pre_25_2.model',
                'ckpt_pre_25_3.model','ckpt_pre_30_0.model','ckpt_pre_30_1.model','ckpt_pre_30_2.model','ckpt_pre_30_3.model',
                'ckpt_pre_35_0.model','ckpt_pre_35_1.model','ckpt_pre_35_2.model','ckpt_pre_35_3.model']
                
            if(i==1):
                model_name_list = ['ckpt_pre_20_cbow_0.model','ckpt_pre_25_cbow_0.model','ckpt_pre_25_cbow_1.model',
                'ckpt_pre_30_cbow_0.model','ckpt_pre_30_cbow_1.model','ckpt_pre_35_cbow_0.model','ckpt_pre_35_cbow_1.model']
            for i in range(len(model_name_list)):
                model = torch.load(os.path.join(model_dir, model_name_list[i]))
                outputs_te = testing(batch_size, test_loader, model, device, True)
                outputs_te = np.reshape(outputs_te,(-1,len(outputs_te))).T
                output_test.append(outputs_te)

        outputs = np.concatenate(output_test[:],axis=1)
        outputs = outputs - 0.5
        
        test_loader = torch.utils.data.DataLoader(dataset = torch.from_numpy(outputs).float(),batch_size = batch_size,shuffle = False,num_workers = 8)
        model = torch.load(os.path.join(model_dir, 'ckpt_ensemble.model'))
        outputs = testing_ens(batch_size, test_loader, model, device, False)
        for i in range(len(outputs)):
            if outputs[i]>=0.5:
                outputs[i] = 1
            else:
                outputs[i] = 0
        tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
        print("save csv ...")
        tmp.to_csv(os.path.join(path_prefix, output_file), index=False)
        print("Finish Predicting")
