import os 
import matplotlib.pyplot as plt 
import numpy as np 
import shutil
import matplotlib
#import theano
from numpy import *
from scipy.misc import imresize

# KERAS IMPORT 
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD, RMSprop, adam
# from keras.utils import np_utils

# SKLEARN IMPORT
# from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split 

# INPUT IMAGE DIMENSIONS
image_rows, image_columns = 200, 200

# NUMEBR OF CHANNELS
image_channels = 1

# PATH WHERE THE ORIGINAL IMAGES ARE STORED 
original_images_path = "C:\\NextGenChatbot\\input_data"
# PATH WHERE THE ORIGINAL IMAGES ARE RESIZED TO 200 x 200 AND STORED 
resized_image_path = "C:\\NextGenChatbot\\input_data_resized"

listing_original_images = os.listdir(original_images_path)
num_original_samples = size(listing_original_images)
print("Total number of original images: {}".format(num_original_samples))
listing_resized_images = os.listdir(resized_image_path)
num_resized_samples = size(listing_resized_images)
print("Total number of resized images: {}".format(num_resized_samples))

def imcrop_tosquare(img):
    if img.shape[0] > img.shape[1]:
        extra = (img.shape[0] - img.shape[1])
        if extra % 2 == 0:
            crop = img[extra // 2:-extra // 2, :]
        else:
            crop = img[max(0, extra // 2 + 1):min(-1, -(extra // 2)), :]
    elif img.shape[1] > img.shape[0]:
        extra = (img.shape[1] - img.shape[0])
        if extra % 2 == 0:
            crop = img[:, extra // 2:-extra // 2]
        else:
            crop = img[:, max(0, extra // 2 + 1):min(-1, -(extra // 2))]
    else:
        crop = img
    return crop

def imcrop(img, amt):
    if amt <= 0 or amt >= 1:
        return img
    row_i = int(img.shape[0] * amt) // 2
    col_i = int(img.shape[1] * amt) // 2
    return img[row_i:-row_i, col_i:-col_i]

def plot_image(filename):
    img = plt.imread(os.path.join('input_data', filename))
    plt.imshow(img)

files = os.listdir('C:\\NextGenChatbot\\input_data')
files = [file for file in files if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file)]

f = files[0]
print(plot_image(f))

img = plt.imread(os.path.join('input_data', f))
squared_image = imcrop_tosquare(img)
crop = imcrop(squared_image, 0.2)
rsz = imresize(crop, (500, 500))
plt.imshow(rsz)





 ############## THIS PART IS ONLY USED FOR RENAMING THE IMAGES FOR EASY IDENTIFICATION ####################

def copy_rename(old_file_name, new_file_name):
	destination_directory = os.path.join('highlight_damage_renamed')
	source_file = os.path.join('highlight_damage', old_file_name)
	shutil.copy(source_file, destination_directory)

	destination_file = os.path.join(destination_directory, old_file_name)
	new_destination_file = os.path.join(destination_directory, new_file_name)
	os.rename(destination_file, new_destination_file)

# this code is for one time use to renamed the images 
# files = os.listdir('highlight_damage')
# for i, filename in enumerate(files):
# 	if ('.jpg' in filename) or ('.jpeg' in filename) or ('.png' in filename):
# 		old_file_ext = None
# 		if '.jpg' in filename:
# 			old_file_ext = '.jpg'
# 		if '.jpeg' in filename:
# 			old_file_ext = '.jpeg'
# 		if '.png' in filename:
# 			old_file_ext = '.png'
# 		new_file_name = 'highlight_damage' + str(i) + str(old_file_ext)
# 		print(new_file_name, filename)
# 		copy_rename(filename, new_file_name)



