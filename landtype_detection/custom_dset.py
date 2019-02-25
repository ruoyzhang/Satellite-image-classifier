import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models, transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.optim import Adam
import math

class custom_dset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir : path to images folder
        label_csv : path to HAM10000_metadata.csv
        """

        # getting the sub folders and getting rid off irrelevant readings
        sub_paths = [os.path.join(data_dir, file_dir) for file_dir in os.listdir(data_dir)]
        sub_paths = [path for path in sub_paths if os.path.isdir(path)]

        # creating a dict to convert string classes into integers that can be converted to tensors
        class_dict = {'water':1, 'trees':2, 'road':3, 'barren_land': 4, 'building': 5, 'grassland':6}

        # creating a dict for all files stored in the different class specific folders
        # the dict contains key-value pairs of the form: full_file_dir: class
        all_files_dict = {}
        for path in sub_paths:
            all_files_dict = {**all_files_dict, **{os.path.join(path, file_name):class_dict[os.path.split(path)[1]] for file_name in os.listdir(path)}}


        # creating a list of all files and store as a class var
        self.all_files = sorted(list(all_files_dict.keys()))

        # setting the all_file_dict to a class variable
        self.dir_to_class_dict = all_files_dict

        # setting the len variable
        self.len = len(self.dir_to_class_dict)
        
        # setting the image transform class method
        if tranform is None:
            normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            tranform = transforms.Compose([
                transforms.Resize(299),
                transforms.CennterCrop(299),
                transforms.ColorJitter(hue = .05, saturation = .05),
                transforms.RandomHorizontalFlip(),
                tramsforms.RandomVerticalFlip(),
                transforms.RandomRotation(360, resample = Image.BILINEAR),
                transforms.ToTensor(),
                normalize,
                ])
        self.tranform = transform

    # the __len__ method
    def __len__(self):
        return(len(self.all_files))

    # the __getitem__ method
    def __getitem__(self, idx):
        file_name = self.all_files[idx]
        image = Image.open(file_name)

        if self.transform:
            image = self.transform(image)

        label = self.dir_to_class_dict[file_name]
        label_t = tensor.from_from_numpy(np.array(label))
        
        return (img_t, label_t)


def train_val_test_split(dataset, train_split, val_split, test_split):
    """
    Split data set into training, validation, and test sets.
    """
    if train_split + val_split + test_split != 1:
        print('Incorrect split sizes')
    
    # Size of data set
    N = dataset.__len__()
    
    # Size of train set
    train_size = math.floor(train_split * N)
    
    # Size of validation set
    val_size = math.floor(val_split * N)
    
    # List of all data indices
    indices = list(range(N))
    
    # Random selection of indices for train set
    train_ids = np.random.choice(indices, size=train_size, replace=False)
    train_ids = list(train_ids)
    
    # Deletion of indices used for train set
    indices = list(set(indices) - set(train_ids))
    
    # Random selection of indices for validation set
    val_ids = np.random.choice(indices, size=val_size, replace=False)
    val_ids = list(val_ids)
    
    # Selecting remaining indices for test set
    test_ids = list(set(indices) - set(val_ids))
    
    # Creating subsets
    train_data = torch.utils.data.Subset(dataset, train_ids)
    val_data = torch.utils.data.Subset(dataset, val_ids)
    test_data = torch.utils.data.Subset(dataset, test_ids)
    
    return train_data, val_data, test_data