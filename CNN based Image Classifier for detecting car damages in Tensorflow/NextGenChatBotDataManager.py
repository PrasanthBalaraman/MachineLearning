import os 
import matplotlib.pyplot as plt 
import numpy as np 
import shutil
import matplotlib
from numpy import *
from scipy.misc import imresize
from PIL import Image
from NextGenChatBotHelper import * 

from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split

# INPUT IMAGE DIMENSIONS
image_rows, image_columns = 500, 500

# NUMEBR OF CHANNELS
image_channels = 1

# PATH WHERE THE ORIGINAL IMAGES ARE STORED 
original_images_path = "C:\\NextGenChatbot\\input_data"
# PATH WHERE THE ORIGINAL IMAGES ARE RESIZED TO 200 x 200 AND STORED 
resized_image_path = "C:\\NextGenChatbot\\input_data_resized"


listing_original_images = os.listdir(original_images_path)
num_original_samples = size(listing_original_images)
#print("Total number of original images: {}".format(num_original_samples))
listing_resized_images = os.listdir(resized_image_path)
num_resized_samples = size(listing_resized_images)
#print("Total number of resized images: {}".format(num_resized_samples))

# Open one image to get the size
image_list = os.listdir(resized_image_path) 
image1 = array(Image.open(resized_image_path+'\\'+image_list[0]))
# get the size of the image 
m, n = image1.shape[0:2]
# get number of images 
images_number = len(image_list)

# create matrix to store all flattened images 
image_matrix = array([array(Image.open(resized_image_path+"\\"+image).convert('RGB')) for image in image_list])
label = np.ones((num_resized_samples,), dtype=int)
label[0:53] = 0
label[53:193] = 1
label[193:] = 2
data, Label = shuffle(image_matrix, label, random_state=2)
train_data = [data, Label]

# splitting the training and test data 
(images, labels) = (train_data[0], train_data[1])
training_images, test_images, training_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=4)
training_one_hot_labels = one_hot_encoder(training_labels, 3)
test_one_hot_labels = one_hot_encoder(test_labels, 3)













