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


def train(data_dir, save_dir, num_class, num_epoch = 20,\
	bs = 4, lr = 1e-3, use_cuda = False, num_workers = 1,\
	name = 'model', train_prop = 0.7, val_prop = 0.2,
	step_size = 4, gamma = 0.1):
	"""
	params:
	data_dir: where the image folder is stored
	save_dir: where the model should be saved after/during training
	number_class: the number of classes to predict
	num_epoch(optional, 20 by default): the number of epochs to train
	bs(optional, 4 by default): the batch size
	lr(optional, 0.001 by default): the starting learning rate
	use_cude(optional, false by default): boolean, wether to use the GPU
	num_workers(optional, 1 by dfault): the number of workers to use for the computation,
				note that if use_cude = True, it should be set to equal 1
	name (optional, 'model' by default): name of the model when it is saved
	train_prop(optional, 0.7 by default): the propotion of the data used for trainning
	val_prop(optional, 0.2 by default): the propotion of the data used for validation
	step_size (4 by default): the frequency for learning rate decay
	gamma (0.1 by default): the factor by which the learning rate decays
	"""

	# checkpoint beginning time
	begin = time.time()

	# instantiate the vgg model
	model = pretrained_inception_v3(num_class, use_cuda)

	# we check if the save_dir exists, if not, we create it
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	# define the model path
	modelpath = os.path.join(save_dir, '{}.pt'.format(name))

	# in the case of paused training, do we wish to continue training?
	if use_cuda:
		model = model.cuda()

	# setting up the loss and accuracy variables
	loss_record = {'train': np.zeros(num_epoch), 'val': np.zeros(num_epoch)}
	acc_record = {'train': np.zeros(num_epoch), 'val': np.zeros(num_epoch)}

	# setting up the loss function and optimisation method
	# we use the cross entropy for the multiclass classification task
	loss_fun = torch.nn.CrossEntropyLoss(reduction = 'sum')
	# we use stochastic gradient descend for optimisation
	optim = SGD(model.parameters(), lr = lr, momentum = 0.9)
	# we apply learning rate decay
	exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size = step_size, gamma = gamma)

	# split the dataset into train, val and test
	test_prop = 1 - train_prop - val_prop
	train_data, val_data, test_data = train_val_test_split(data_dir, train_prop, val_prop, test_prop)

	# saving the test data. We save the test data first in case we want to terminate the training early
	with open('{}.pickle'.format(os.path.join(save_dir, 'test_data')), 'wb') as handle:
		pickle.dump(test_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

	# set up the datasets
	dset = {'train': custom_dset(data_files = train_data, transform = 'train'),
			'val': custom_dset(data_files = val_data, transform = 'val')}

	# setting up a random weighted sampler to ensure the classes are balanced in the training phase
	# calculating weights for the dset indices according to their respective class
	train_class_sample_count = Counter(dset['train'].labels)
	sorted_train_class_sample_count = [train_class_sample_count[key] for key in sorted(train_class_sample_count.keys())]
	weights = 100000./torch.tensor(sorted_train_class_sample_count, dtype = torch.float)
	samples_weights = [weights[label] for label in dset['train'].labels]
	# the sampler
	sampler = WeightedRandomSampler(weights=samples_weights,
									num_samples=len(samples_weights),
									replacement=True)
	# the dataloaders, we create 2 dataloaders for the train and val phase seperately
	dataloaders = {'train': DataLoader(dset['train'], batch_size = bs, sampler = sampler, num_workers = num_workers, pin_memory = False),
					'val': DataLoader(dset['val'], batch_size = bs, shuffle = True, num_workers = num_workers, pin_memory = False)}

	# create variables for storing best performing weights during training
	best_model_wts = model.state_dict()
	best_acc = 0.0

	# iterate through the training epochs
	for epoch in range(num_epoch):
		print('Epoch {}/{}'.format(epoch, num_epoch - 1))
		print('-' * 10)

		# each epoch will have a training and validation phase
		for phase in ['train', 'val']:

			# recording the running performance and the dataset size
			running_loss = 0.0
			running_corrects = 0
			size = 0

			if phase == 'train':
				exp_lr_scheduler.step()
				# setting model to trainning mode
				model.train()
			else:
				# setting model to validation mode
				model.eval()

			# now we iterate over the data
			for inputs, labels in tqdm(dataloaders[phase]):
				# counting how many images is contained in this batch
				batch_count = labels.size(0)
				# if GPU is used, cudafy inputs
				if use_cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()

				# zero the parameter gradients
				optim.zero_grad()

				# feed inputs into the model
				output = model(inputs)


				# calculate the loss and prediction performance statistics
				if type(output) == tuple:
					output, _ = output
				_, preds = torch.max(output.data, 1)
				loss = loss_fun(output, labels)
				running_loss += loss
				running_corrects += preds.eq(labels.view_as(preds)).sum()

				# backprop and optimise if in training stage
				if phase == 'train':
					loss.backward()
					optim.step()
				
				# update dataset size
				size += batch_count

			epoch_loss = running_loss / size
			epoch_acc = running_corrects.item() / size

			# recording the historical performance
			loss_record[phase][epoch] = epoch_loss
			acc_record[phase][epoch] = epoch_acc

			print('{} loss: {:.4F} Acc: {:.4F}'.format(phase, epoch_loss, epoch_acc))

			# deep copy and save the model if best performance
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
				torch.save(model.state_dict(), os.path.join(save_dir, '().pt'.format(name)))

	# saving the record performances
	with open('{}.pickle'.format(os.path.join(save_dir, name + '_loss_performances')), 'wb') as handle:
		pickle.dump(loss_record, handle, protocol = pickle.HIGHEST_PROTOCOL)
	with open('{}.pickle'.format(os.path.join(save_dir, name + '_acc_performances')), 'wb') as handle:
		pickle.dump(acc_record, handle, protocol = pickle.HIGHEST_PROTOCOL)

	# calculating time elapsed
	time_elapsed = time.time() - begin
	print('Training complete in {:.0F}m {:0F}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best validation acc: {:.4F}'.format(best_acc))

	# load the best model weights
	model.load_state_dict(best_model_wts)
	return(loss_record, acc_record, model, test_data)