# test.py
# 這個 block 用來對 testing_data.txt 做預測
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device, origin):
    model.eval()
    ret_output = []
    ret_origin_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            ret_origin_output += outputs.float().tolist()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
    if(origin):
        return ret_origin_output
    else:
        return ret_output

def testing_ens(batch_size, test_loader, model, device, origin):
    model.eval()
    ret_output = []
    ret_origin_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            ret_origin_output += outputs.float().tolist()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
    if(origin):
        return ret_origin_output
    else:
        return ret_output