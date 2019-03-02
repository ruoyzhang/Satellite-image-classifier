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


#----------------------------------------------------
# the pretrained inception v3 model class
#----------------------------------------------------

class pretrained_inception_v3(nn.Module):

	def __init__(self, num_class, use_cuda):
		super(pretrained_inception_v3, self).__init__()
		"""
		num_class : the total number of classes to predict
		use_cude : if we're using the GPU to compute and optimise
		"""

		# setting variables for if we're using the GPU, the number of output classes as well as the tensor dtype
		self.use_cuda = use_cuda
		self.num_class = num_class
		self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

		# we're using the pretrained inceptionv3 model provided by torchvision
		model = models.inception_v3(pretrained=True)
		self.model = model.cuda() if self.use_cuda else model

		# freeze all layer weights
		for param in self.model.parameters():
			param.requires_grad = False

		# modifying the classifier layer
		num_features = self.model.fc.in_features
		self.model.fc = nn.Linear(num_features, num_class)

		# unfreeze the params for the classifier layer
		for param in self.model.fc.parameters():
			param.requires_grad = True

		# we also choose to unfreeze the weights of the Conv2d_4a_3x3 layer
		for param in inceptionv3.Conv2d_4a_3x3.parameters():
			param.requires_grad = True

	def forward(self, inputs):
		return(self.model(inputs))


