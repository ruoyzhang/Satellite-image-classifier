import random
from scipy import ndarray
import skimage as sk
from skimage import util
from math import floor
import cv2
from tqdm import tqdm
import numpy as np


def random_rotation_25(image_array: ndarray):
    # pick a random degree of rotation between 25 on the left and 25 degrees on the right
    random_degree = random.uniform(-25, 25)
    return(sk.transform.rotate(image_array, random_degree))

def random_rotation_75(image_array: ndarray):
    # pick a random degree of rotation between 26 on the right and 75 degrees on the right
    random_degree = random.uniform(26, 75)
    return(sk.transform.rotate(image_array, random_degree))

def random_rotation_90(image_array: ndarray):
    # randomly rotate image of 90 degrees either to the left or to the right
    random_degree = random.choice(-90, 90)
    return(sk.transform.rotate(image_array, random_degree))

def random_noise(image_array: ndarray):
    # add random noise to the image
    return(sk.util.random_noise(image_array))

def horizontal_flip(image_array: ndarray):
    # horizontal flip
    return(image_array[:, ::-1])

def vertical_flip(image_array: ndarray):
    # vertical flip
    return(image_array[::-1, :])

def transpose(image_array: ndarray):
	# transpose the image
	return(image_array[::-1, ::-1])

def zoom(image_array: ndarray):
	# zoom in on the image, maximum zoom: 1.4
	dim = image_array.shape[0]
	zoom_factor = random.uniform(1.01, 1.4)
	zoomed_image = sk.transform.rescale(image_array, zoom_factor)
	crop_border = floor((zoomed_image.shape[0] - dim)/2)
	cropped_image = zoomed_image[crop_border : crop_border + dim, crop_border : crop_border + dim]
	return(cropped_image)

def image_augmentation(image_dirs, fold):
	"""
	this function will augment selected images and return it as a vector
	-----
	image_dirs: list of dirs to the images to transform
	fold: the number of times the data is augmented, max = 8
	"""

	# check if fold limits are respected
	if fold > 8 or fold < 1:
		return('fold has to be between 1 and 8')

	# convert it to an integer in case where a float was received
	fold = int(fold)

	# establish all the functions to use for augmentation
	function_list = [random_rotation_25, random_rotation_75, random_rotation_90, random_noise, horizontal_flip, vertical_flip, transpose, zoom]

	# randomly choose which augmentations to use:
	augmentations_to_use = random.sample(function_list, fold)

	# print augmentation functions to use
	print('the functions to be used for augmentation are: ')
	for fun in augmentations_to_use:
		print(str(fun).split(' ')[1])

	# create a list to store results
	augmented = []

	# now we augment:
	for img_dir in tqdm(image_dirs):
		# first we read the images
		img = cv2.imread(img_dir)
		b, g, r = cv2.split(img)
		rgb_img = cv2.merge([r, g, b])

		# augment the image
		for aug in augmentations_to_use:
			aug_img = aug(rgb_img)
			aug_img.shape = (1, 3*28*28)
			augmented.append(aug_img)

	# convert to a nparray
	aug_img = np.array(aug_img)

	# reshape the array
	aug_img.shape = (aug_img.shape[0], aug_img.shape[2])

	return(aug_img)





