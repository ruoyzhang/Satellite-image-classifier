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
from torch.optim import 
from torch.optim import Adam, SGD, lr_scheduler
import math
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def train(data_dir, save_dir, num_class,\
	epoch = 20, bs = 4, lr = 1e-3, use_cuda = False,\
	num_workers = 1, save_freq = 5, name = 'model',\
	train_prop = 0.7, val_prop = 0.2, conti = False):
	"""
	params:
	data_dir: where the image folder is stored
	save_dir: where the model should be saved after/during training
	number_class: the number of classes to predict
	epoch(optional, 20 by default): the number of epochs to train
	bs(optional, 4 by default): the batch size
	lr(optional, 0.001 by default): the starting learning rate
	use_cude(optional, false by default): boolean, wether to use the GPU
	num_workers(optional, 1 by dfault): the number of workers to use for the computation,
				note that if use_cude = True, it should be set to equal 1
	save_freq(optional, 5 by default): the model will be saved every set number of epochs
	name (optional, 'model' by default): name of the model when it is saved
	train_prop(optional, 0.7 by default): the propotion of the data used for trainning
	val_prop(optional, 0.2 by default): the propotion of the data used for validation
	conti(option, False by default): boolean, in the case of paused training, do we wish to continue from a saved model?
	"""

	# we check if the save_dir exists, if not, we create it
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	# define the model path
	modelpath = os.path.join(save_dir, '{}.pt'.format(name))

	# in the case of paused training, do we wish to continue training?
	if os.path.isfile(modelpath) and conti:
		model.load_state_dict(torch.load(modelpath))
	if use_cuda:
		model = model.cuda()

	# setting model to train mode
	model.train()

	# setting up the training loss and accuracy variables
	loss_train = np.zeros(epoch)
	acc_train = np.zeros(epoch)

	# setting up the loss function and optimisation method
	# we use the cross entropy for the multiclass classification task
	loss_fun = torch.nn.CrossEntropyLoss(reduction = 'sum')
	# we use stochastic gradient descend for optimisation
	optim = SGD(model.paramters(), lr = lr, momentum = 0.9)
	# we apply learning rate decay
	exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size = 7, gamma = 0.1)


	# set up the dataset
	dataset = custom_dset(data_dir = data_dir)

	# setting up a random weighted sampler to ensure the classes are balanced in the training phase
	# calculating weights for the dset indices according to their respective class
	class_sample_counts = Counter(dset.labels)
	sorted_class_sample_count = [class_sample_count[key] for key in sorted(class_sample_count.keys())]
	weights = 100000./torch.tensor(sorted_class_sample_count, dtype = torch.float)
	samples_weights = [weights[label] for label in dset.labels]
	# the sampler
	sampler = WeightedRandomSampler(weights=samples_weights,
									num_samples=len(samples_weights),
									replacement=True)

	# split the dataset into train, val and test
	test_prop = 1 - train_prop - val_prop
	train_data, val_data, test_data = train_val_test_split(dataset, train_prop, val_prop, test_prop)















