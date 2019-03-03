import random
from scipy import ndarray
import skimage as sk
from skimage import util
from math import floor


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
	dim = image.array.shape[0]
	zoom_factor = random.uniform(1.01, 1.4)
	zoomed_image = sk.transform.rescale(image_array, zoom_factor)
	crop_border = floor((zoomed_image.shape[0] - dim)/2)
	cropped_image = zoomed_image[crop_border : crop_border + dim, crop_border : crop_border + dim]
	return(cropped_image)

