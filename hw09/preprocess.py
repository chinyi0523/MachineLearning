import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    """
    image_list = np.transpose(image_list, (1, 0, 2, 3))
    #print(np.mean(image_list,axis=(1,2,3)))
    #print(np.std(image_list,axis=(1,2,3)))
    for i in range(3):
        print(image_list[i].shape)
        image_list[i] = (image_list[i] - np.mean(image_list,axis=(1,2,3))[i])/np.std(image_list,axis=(1,2,3))[i]
    image_list = np.transpose(image_list, (1, 0, 2, 3))
    #print(np.mean(image_list,axis=(0,2,3)))
    #print(np.std(image_list,axis=(0,2,3)))
    """
    return image_list