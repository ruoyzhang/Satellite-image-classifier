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


def train_val_test_split(data_dir, train_split, val_split, test_split):
    """
    Split data set into training, validation, and test sets.
    data_dir : path to images folder
    train_split : proportion of the data to be used for training
    val_split : proportion of the data to be used for validation
    test_split : proportion of the data to be used for testing
    """

    # getting the sub folders and getting rid off irrelevant readings
    sub_paths = [os.path.join(data_dir, file_dir) for file_dir in os.listdir(data_dir)]
    sub_paths = [path for path in sub_paths if os.path.isdir(path)]

    # creating a dict to convert string classes into integers that can be converted to tensors
    class_dict = {'water':0, 'trees':1, 'road':2, 'barren_land': 3, 'building': 4, 'grassland':5}

    # creating a dict for all files stored in the different class specific folders
    # the dict contains key-value pairs of the form: full_file_dir: class
    all_files_dict = {}
    for path in sub_paths:
        all_files_dict = {**all_files_dict, **{os.path.join(path, file_name):class_dict[os.path.split(path)[1]] for file_name in os.listdir(path)}}

    # now sample according to the proportions
    # Size of data set
    N = len(all_files_dict)
    
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

    # creating subsets in the forms of dictionaries
    # these dicts contain key-value pairs of the form 'file_dir : class'
    all_files = sorted(list(all_files_dict.keys()))
    train_files = {all_files[i]: all_files_dict[all_files[i]] for i in train_ids}
    val_files = {all_files[i]: all_files_dict[all_files[i]] for i in val_ids}
    test_files = {all_files[i]: all_files_dict[all_files[i]] for i in test_ids}

    return(train_files, val_files, test_files)








class custom_dset(Dataset):
    def __init__(self, data_files, transform):
        """
        data_files : one of the outputs of the train_val_test_split function
                     a dictionary containing keys of image directories and values of their respective classes
        transform : either 'train' or 'val'
                    indicates whether the dataset is for training or validation purposees
        """

        # setting the all_file_dict to a class variable
        self.dir_to_class_dict = data_files

        # setting all file directories to a class variable
        self.all_files = sorted(list(data_files.keys()))

        # creating a list of all classes in the order of the all_files class variable
        self.labels = [data_files[key] for key in self.all_files]

        # setting the len variable
        self.len = len(self.dir_to_class_dict)
        
        # setting the image transform class method
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        if transform == 'train':
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ColorJitter(hue = .05, saturation = .05),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90, resample = Image.BILINEAR),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(299),
                transform.ToTensor()])
        self.transform = transform

    # the __len__ method
    def __len__(self):
        return(len(self.all_files))

    # the __getitem__ method
    def __getitem__(self, idx):
        file_name = self.all_files[idx]
        image = Image.open(file_name)
        # the images are in .png format, which comes with an extra alpha channel
        # since we're using a pretrained model,the data has to be converted to 3 channels
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.dir_to_class_dict[file_name]
        label_t = torch.from_numpy(np.array(label))
        
        return (image, label_t)
