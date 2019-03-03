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
from torch.optim import Adam, SGD, lr_scheduler
import math
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import time
from pretrained_inceptionv3 import pretrained_inception_v3
from custom_dset_new import custom_dset, train_val_test_split
import pickle


def test(model, test_files, bs):
	"""
	model : the model to be tested
	test_files : the directories to the test images, output of the train_val_test_split function
	bs : batch size

	"""

	# set model to eval mode
	model.eval()

	# recording the running performance and the dataset size
	running_loss = 0.0
	running_corrects = 0
	size = 0

	# set up the loss function
	loss_fun = torch.nn.CrossEntropyLoss(reduction = 'sum')

	# set up the datasets
	dset = custom_dset(data_files = test_files, transform = 'val')

	# set up the dataloader
	dataloader = DataLoader(dset, batch_size = bs, shuffle = True, num_workers = 1, pin_memory = False)

	# now iterate over the images to make predictions
	for inputs, labels in tqdm(dataloader):
		# counting how many images is contained in this batch
		batch_count = labels.size(0)
		# Forward pass
		outputs = model(inputs)

		# calculate the loss and prediction performance statistics
		if type(output) == tuple:
			output, _ = output
			_, preds = torch.max(output.data, 1)
			loss = loss_fun(output, labels)
			running_loss += loss
			running_corrects += preds.eq(labels.view_as(preds)).sum()

	
		# update dataset size
		size += batch_count

	# compute the model's performance
	model_loss = running_loss / size
	model_acc = running_corrects / size

	return(model_loss, model_acc)
