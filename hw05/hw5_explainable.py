# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
from torch.autograd import Variable
from torch.optim import SGD
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

"""## Dataset definition and creation"""

# 助教 training 時定義的 dataset
# 因為 training 的時候助教有使用底下那些 transforms，所以 testing 時也要讓 test data 使用同樣的 transform
# dataset 這部分的 code 基本上不應該出現在你的作業裡，你應該使用自己當初 train HW3 時的 preprocessing
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        train_Transform = transforms.Compose([
          transforms.Resize(size=(128, 128)),
          transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
          transforms.RandomRotation(15), #隨機旋轉圖片
          transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)      
        ])
        eval_Transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),                                    
        transforms.ToTensor(),
        
        ])
        self.transform = train_Transform if mode == 'train' else eval_Transform

    # 這個 FoodDataset 繼承了 pytorch 的 Dataset class
    # 而 __len__ 和 __getitem__ 是定義一個 pytorch dataset 時一定要 implement 的兩個 methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

# 給予 data 的路徑，回傳每一張圖片的「路徑」和「class」
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
  model.eval()
  x = x.cuda()

  # 最關鍵的一行 code
  # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
  # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
  x.requires_grad_()

  y_pred = model(x)
  loss_func = torch.nn.CrossEntropyLoss()
  loss = loss_func(y_pred, y.cuda())
  loss.backward()

  saliencies = x.grad.abs().detach().cpu()
  # saliencies: (batches, channels, height, weight)
  # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
  # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
  # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗，
  # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
  # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
  saliencies = torch.stack([normalize(item) for item in saliencies])
  return saliencies

######################


def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
  # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
  # cnnid, filterid: 想要指定第幾層 cnn 中第幾個 filter
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output

  hook_handle = model.cnn[cnnid].register_forward_hook(hook)
  # 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義的 function 後才可以繼續 forward 下一層 cnn
  # 因此上面的 hook function 中，我們就會把該層的 output，也就是 activation map 記錄下來，這樣 forward 完整個 model 後我們就不只有 loss
  # 也有某層 cnn 的 activation map
  # 注意：到這行為止，都還沒有發生任何 forward。我們只是先告訴 pytorch 等下真的要 forward 時該多做什麼事
  # 注意：hook_handle 可以先跳過不用懂，等下看到後面就有說明了

  # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
  model(x.cuda())
  # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
  # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor

  # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
  x = torch.rand(1, 3, 128, 128).cuda()
  # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
  x.requires_grad_()
  # 我們要對 input image 算偏微分
  optimizer = Adam([x], lr=lr)
  # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)

    objective = -layer_activations[:, filterid, :, :].abs().sum()
    # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
    # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
    # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization

    objective.backward()
    # 計算 filter activation 對 input image 的偏微分
    optimizer.step()
    # 修改 input image 來最大化 filter activation
  filter_visualization = x.detach().cpu().squeeze()
  # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

  hook_handle.remove()
  # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
  # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
  # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

  return filter_activations, filter_visualization

###############
def predict(input):
  # input: numpy array, (batches, height, width, channels)

  model.eval()
  input = torch.FloatTensor(input).permute(0, 3, 1, 2)
  # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
  # 也就是 (batches, channels, height, width)

  output = model(input.cuda())
  return output.detach().cpu().numpy()

def segmentation(input):
  # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
  return slic(input, n_segments=100, compactness=1, sigma=1)

def label2string(label):
  lab = str(int(label))
  return {
    '0': "Bread",
    '1': "Dairy product",
    '2': "Dessert",
    '3': "Egg", 
    '4': "Fried food", 
    '5': "Meat", 
    '6': "Noodles/Pasta",
    '7': "Rice",
    '8': "Seafood",
    '9': "Soup",
    '10':"Vegetable/Fruit"
    }.get(lab,'Error')
  
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
def get_gradients(net_in, net, layer):     
  net_in = net_in.unsqueeze(0).cuda()
  net_in.requires_grad = True
  net.zero_grad()
  hook = Hook(layer)
  net_out = net(net_in)
  loss = hook.output[0].norm()
  loss.backward()
  return net_in.grad.data.squeeze()
def dream(image, net, layer, iterations, lr):
  #image_tensor = transforms.ToTensor()(image)
  #image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).cuda()
  image_tensor = image.cuda()
  #image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).cuda()
  for i in range(iterations):
    gradients = get_gradients(image_tensor, net, layer)
    image_tensor.data = image_tensor.data + lr * gradients.data

  img_out = image_tensor.detach().cpu()
  #img_out = denorm(img_out)
  img_out_np = img_out.numpy().transpose(1,2,0)
  img_out_np = np.clip(img_out_np, 0, 1)
  img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
  return img_out_pil

if __name__ == '__main__':

  #python3 hw5_explainable.py -Saliency ./food-11 ./output
  if(len(sys.argv)!=4):
    print("Usage  : python3 hw5_explainable.py <option> <dataset directory> <output directory>")
    print("Example: python3 hw5_explainable.py -Saliency ./food-11 ./output")
    print("Options: -Saliency -Filter -Lime -Heatmap")
    exit()

  option = sys.argv[1]
  
  args = {
        'ckptpath': './model_VGG.pkl',
        'dataset_dir': sys.argv[2],
        'output_prefix': sys.argv[3]
  }
  args = argparse.Namespace(**args)
  
  model = Classifier().cuda()
  checkpoint = torch.load(args.ckptpath)
  model.load_state_dict(checkpoint)
      
  train_paths, train_labels = get_paths_labels(os.path.join(args.dataset_dir, 'training'))
  train_set = FoodDataset(train_paths, train_labels, mode='eval')

  if(option == "-Saliency"):
    print("Generating Saliency Maps ...")
    img_indices = []
    
    img_indices.append([83, 426, 523, 798]) #0-993
    img_indices.append([1704, 1858, 2000, 2123]) #994-1422
    img_indices.append([2546, 2674, 3065, 3472]) #2132-3631
    img_indices.append([3826, 4218, 4436, 4594]) #3632-4617
    img_indices.append([4707, 4890, 5105, 5367]) #4618-5465
    img_indices.append([5679, 6000, 6246, 6666]) #5466-6790
    img_indices.append([6939, 6895, 7012, 7120]) #6791-7230
    img_indices.append([7345, 7378, 7465, 7501]) #7231-7511
    img_indices.append([7647, 7936, 8217, 8273]) #7512-8366
    img_indices.append([8432, 8590, 8598, 9643]) #8367-9866
    img_indices.append([1478, 1546, 1325, 1643]) #1423-2131
    img_indices.append([83, 4218, 4707, 8598])
    img_indices.append([1546, 2000, 6000, 8590])
    for i in range(len(img_indices)):
      images, labels = train_set.getbatch(img_indices[i])
      saliencies = compute_saliency_maps(images, labels, model)

      fig, axs = plt.subplots(2, len(img_indices[i]), figsize=(15, 8))
      for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
          axs[row][column].imshow(img.permute(1, 2, 0).numpy())
      if i <11:
        fig.suptitle(label2string(labels[0]))    
      plt.savefig(os.path.join(args.output_prefix, 'saliency_map{iter}.png'.format(iter = i+1)))
  
  if(option == "-Filter"):
    
    layer_activations = None
    filterindex=[]
    _filter = [10,50,[83, 426, 523, 798],[1704, 1858, 2000, 2123],[2546, 2674, 3065, 3472],[3826, 4218, 4436, 4594], [4707, 4890, 5105, 5367], [5679, 6000, 6246, 6666],[6939, 6895, 7012, 7120],[7345, 7378, 7465, 7501],[7647, 7936, 8217, 8273],[8432, 8590, 8598, 9643],[1478, 1546, 1325, 1643]]
    filterindex.append(_filter)
    _filter = [10,0,[83, 426, 523, 798],[1704, 1858, 2000, 2123],[2546, 2674, 3065, 3472],[3826, 4218, 4436, 4594], [4707, 4890, 5105, 5367], [5679, 6000, 6246, 6666],[6939, 6895, 7012, 7120],[7345, 7378, 7465, 7501],[7647, 7936, 8217, 8273],[8432, 8590, 8598, 9643],[1478, 1546, 1325, 1643],[83,4218,4707,8598],[1546,2000,6000,8590],[2345,5427,6789,8765]]
    filterindex.append(_filter)
    _filter = [9,21,[83, 426, 523, 798],[1704, 1858, 2000, 2123],[2546, 2674, 3065, 3472],[3826, 4218, 4436, 4594], [4707, 4890, 5105, 5367], [5679, 6000, 6246, 6666],[6939, 6895, 7012, 7120],[7345, 7378, 7465, 7501],[7647, 7936, 8217, 8273],[8432, 8590, 8598, 9643],[1478, 1546, 1325, 1643],[83,4218,4707,8598],[1546,2000,6000,8590]]
    filterindex.append(_filter)
    '''
    [[10, 0, [83, 4218, 4707, 8598], [1546, 2000, 6000, 8590], [2345, 5427, 6789, 8765]], [9, 21, [83, 4218, 4707, 8598], [1546, 2000, 6000, 8590]], [10, 50, [83, 4218, 4707, 8598], [1546, 2000, 6000, 8590]]]
    '''

    for filter_num in range(len(filterindex)):
      print("Generating Filter Explaination Graph with cnn: ",filterindex[filter_num][0],"filter: ",filterindex[filter_num][1],"...")
      for graph in range(len(filterindex[filter_num])-2):
        #img_indices = [2345,5427, 6789, 8765]
        img_indices = filterindex[filter_num][graph+2]
        images, labels = train_set.getbatch(img_indices)
        filter_activations, filter_visualization = filter_explaination(images, model, cnnid=filterindex[filter_num][0], filterid=filterindex[filter_num][1], iteration=1000, lr=0.1)
        # (8,3) (8,8)
        # (10,10) [1546,2000,6000,8590] [83, 4218, 4707, 8598] [2674, 3210, 6575, 7654]
        # (10.50) [83, 4218, 4707, 8598] [1546,2000,6000,8590]
        # 畫出 filter visualization
        if(graph==0):
          plt.figure(figsize=(15, 8))
          plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
          plt.savefig(os.path.join(args.output_prefix, '{cnnid}_{filterid}_filter_visualization.png'.format(cnnid=filterindex[filter_num][0], filterid=filterindex[filter_num][1])))
    
        fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
        for i, img in enumerate(images):
          axs[0][i].imshow(img.permute(1, 2, 0))
        for i, img in enumerate(filter_activations):
          axs[1][i].imshow(normalize(img))
        if(label2string(labels[0])==label2string(labels[1])):
          fig.suptitle(label2string(labels[0]))
        plt.savefig(os.path.join(args.output_prefix, '{cnnid}_{filterid}_filter_activation{iter}.png'.format(cnnid=filterindex[filter_num][0], filterid=filterindex[filter_num][1], iter=graph+1)))

  if(option=="-Lime"):
    print("Generating Lime ...")
    img_indices = [] #2 1340  8 705 761
    #993 428 1499 985 847 1324 439 279 854 1499 708
    #img_indices = [1546,2000,6000,8590]
    img_indices.append([83, 426, 523, 798]) #0-993
    img_indices.append([1704, 1858, 2000, 2123]) #994-1422
    img_indices.append([2546, 2674, 3065, 3472]) #2132-3631
    img_indices.append([3826, 4218, 4436, 4594]) #3632-4617
    img_indices.append([4707, 4890, 5105, 5367]) #4618-5465
    img_indices.append([5679, 6000, 6246, 6666]) #5466-6790
    img_indices.append([6939, 6895, 7012, 7120]) #6791-7230
    img_indices.append([7345, 7378, 7465, 7501]) #7231-7511
    img_indices.append([7647, 7936, 8217, 8273]) #7512-8366
    img_indices.append([8432, 8590, 8598, 9643]) #8367-9866
    img_indices.append([1478, 1546, 1325, 1643]) #1423-2131


    for i in range(len(img_indices)):
      images, labels = train_set.getbatch(img_indices[i])
      print(labels)

      fig, axs = plt.subplots(2, 4, figsize=(15, 8))
      np.random.seed(16)
      # 讓實驗 reproducible
      for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation,labels=(label.item(),),top_labels=None)
        # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
        # classifier_fn 定義圖片如何經過 model 得到 prediction
        # segmentation_fn 定義如何把圖片做 segmentation

        lime_img, mask = explaination.get_image_and_mask(label=label.item(),positive_only=False,hide_rest=False,num_features=11,min_weight=0.05)
        # 把 explainer 解釋的結果轉成圖片
        axs[0][idx].imshow(images[idx].permute(1, 2, 0))
        axs[1][idx].imshow(lime_img)
      fig.suptitle(label2string(labels[0]))
      plt.savefig(os.path.join(args.output_prefix,'lime_label{iter}.png'.format(iter=i+1)))
      # 從以下前三章圖可以看到，model 有認出食物的位置，並以該位置為主要的判斷依據
      # 唯一例外是第四張圖，看起來 model 似乎比較喜歡直接去認「碗」的形狀，來判斷該圖中屬於 soup 這個 class
  if(option=="-Dream"):
    print("Generating Deep Dream ...")
    img_indices = []
    img_indices.append([83, 1858, 2674, 4218])
    img_indices.append([4707, 6000, 7034, 7321])
    img_indices.append([8212, 8598, 1023])
    for j in range(len(img_indices)):
      images, labels = train_set.getbatch(img_indices[j])
      layer = list( model.cnn.modules() )[20]
      print(layer)
      print(labels)
      img = []
      for i in range(len(images)):
        img.append(dream(images[i],model,layer,20,1))
      fig, axs = plt.subplots(2, len(img_indices[j]), figsize=(15, 8))
      for i in range(len(images)):
        axs[0][i].imshow(images[i].permute(1, 2, 0))
        axs[0][i].set_title('Origin '+label2string(labels[i]))
      for i in range(len(img)):
        axs[1][i].imshow(img[i])
        axs[1][i].set_title('Dream  '+label2string(labels[i]))
      plt.savefig(os.path.join(args.output_prefix,'dream{iter}.png'.format(iter=j+1)))