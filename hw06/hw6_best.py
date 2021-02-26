import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np
import sys
import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_all = []
        wrong, fail, success = 0, 0, 0
        count = 0
        test_acc = 0
        print("Start Attacking...")
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            test=0
            while(True):
                
                data.requires_grad = True
                # 將圖片丟入 model 進行測試 得出相對應的 class
                output = self.model(data)
                init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
                if init_pred.item() != target.item():
                    if(test==0):
                        test_acc +=1
                    wrong += 1
                    count+=1
                    percent = round(1.0 * count / 2,2)
                    if count==200:
                        print('Current : %s [%d/%d]'%(str(percent)+'%',count,200),end='\n')
                    else:
                        print('Current : %s [%d/%d]'%(str(percent)+'%',count,200),end='\r')
                    new_img = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    new_img = torch.clamp(new_img.squeeze(), 0, 1).detach().cpu()
                    img_trans = transforms.Compose([transforms.ToPILImage()])
                    new_img = img_trans(new_img)
                    new_img.save(os.path.join(output_dir,"{num}.png".format(num="%03d" % (count-1))))
                    break
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
                #if test==138:
                #    epsilon = 0.7
                loss = F.nll_loss(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                data = self.fgsm_attack(data, epsilon, data_grad)
                data = data.detach()
                test+=1
            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(data)
            final_pred = output.max(1, keepdim=True)[1]
            data = data.detach()
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                attack = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                attack = attack.squeeze().detach().cpu().numpy()
                adv_all.append(attack)
                           
        final_acc = (fail / (wrong + success + fail))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))  
        return final_acc,adv_all

    def test(self):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        wrong, fail, success = 0, 0, 0
        count = 0
        test_acc = 0
        print("Start Testing...")
        print("---Adv Fail List---")
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                test_acc +=1  
            else:
                print("No: ",count)
            count +=1
        print("-------------------")
        print('Adversial Attack Success:  %s [%d/%d]'%(str(round(1.0 * test_acc / 2,2))+'%',test_acc,200),end='\n')

if __name__ == '__main__':
    # 讀入圖片相對應的 label
    if len(sys.argv)!=4:
        print("Usage:   python3 hw6_best.py <input directory> <output directory> <option>")
        print("Example: python3 hw6_best.py ./data ./output -attack")
        exit()
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    df = pd.read_csv(os.path.join(input_dir,"labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(input_dir,"categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    if(sys.argv[3]=='-attack'):
        attacker = Attacker(os.path.join(input_dir,"images"), df)
        # 要嘗試的 epsilon
        epsilons = [0.035]
        accuracies, examples, adv_output = [], [], []

        # 進行攻擊 並存起正確率和攻擊成功的圖片
        for eps in epsilons:
            acc, adv = attacker.attack(eps)
            accuracies.append(acc)
            adv_output.append(adv)
        
    if(sys.argv[3]=='-test'):
        attacker = Attacker(os.path.join(input_dir,"images"), df)
        attacker.test()
    exit()


    #for i in range(len(examples)):
        
