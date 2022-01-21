import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
from ast import literal_eval
import numpy as np
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
data.pix2pix_dataset.py
'''

class CustomDataset(Dataset):
    def __init__(self, args):
        
        self.args = args

        self.dataset_path = args.dataset_path
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.channels = args.img_ch
        self.segmap_channel = args.segmap_ch
        self.augment_flag = args.augment_flag # Image augmentation use or not
        self.batch_size = args.batch_size

        self.img_dataset_path = os.path.join(self.dataset_path, 'image')
        self.segmap_dataset_path = os.path.join(self.dataset_path, 'segmap')
        # self.segmap_test_dataset_path = os.path.join(dataset_path, 'segmap_test')

        self.image = glob(self.img_dataset_path + '/*.*')
        self.segmap = glob(self.segmap_dataset_path + '/*.*')
        
        # self.segmap_test = []

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        label_transformer = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        image_transformer = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        segmap = Image.open(self.segmap[idx])
        segmap_tensor = label_transformer(segmap) * 255.0
        segmap_tensor[segmap_tensor == 255] =  self.args.label_nc # label_nc 를 18로 해놓음.

        image = Image.open(self.image[idx])
        image = image.convert('RGB')
        image_tensor = image_transformer(image)

        return image_tensor, segmap_tensor

def imsave(input, name):
    input = input.numpy().transpose((1, 2, 0))
    plt.imshow(input)
    plt.savefig(name)

def load_dataset(args):
    
    train_datasets = CustomDataset(args)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return train_dataloader
