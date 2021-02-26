import re
import torch
from glob import glob
from PIL import Image
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import os
import numpy as np
import pickle

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]

def get_dataloader(datapath,Trans,mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']
    """
    with open('{m}.pkl'.format(m=mode), 'rb') as f:
        dataset = pickle.load(f)
    """
    dataset = MyDataset(
        os.path.join(datapath,mode),
        transform=Trans)
    """
    with open('{m}.pkl'.format(m=mode), 'wb') as f:
        pickle.dump(dataset, f)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

def concatenate_dataloader(datapath,Trans,mode='training', batch_size=32):

    assert mode in ['training']
    datasets=[]
    print("Load train data...")
    dataset_train = MyDataset(
        os.path.join(datapath,'training'),
        transform=Trans)
    print("Load validation data...")
    dataset_val = MyDataset(
        os.path.join(datapath,'validation'),
        transform=Trans)
    datasets.append(dataset_train)
    datasets.append(dataset_val)
    print("Concating...")
    dataset = ConcatDataset(datasets)
    print("Constructing...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader