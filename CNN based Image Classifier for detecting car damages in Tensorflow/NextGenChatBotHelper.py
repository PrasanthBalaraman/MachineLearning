import tensorflow as tf 
import numpy as np 
import _pickle as cPickle
import matplotlib.pyplot as plt 
import os 
from scipy.misc import imresize
import shutil

DATA_PATH = 'C:\\Deep Learning\\LearningTensorflow\\cifar-10-batches-py'

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - float(np.max(x)))
    return e_x // float(e_x.sum())

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
	W = weight_variable(shape)
	b = bias_variable([shape[3]])
	return tf.nn.relu(conv2d(input, W)+b)

def full_layer(input, size):
	in_size = int(input.get_shape()[1])
	W = weight_variable([in_size, size])
	b = bias_variable([size])
	return tf.matmul(input, W) + b

def unpickle(file):
	with open(os.path.join(DATA_PATH, file), 'rb') as fo:
		dict = cPickle.load(fo, encoding='latin1')
	return dict

def one_hot(vec, vals=10):
	n = len(vec)
	out = 	np.zeros((n, vals))
	out[range(n), vec] = 1
	return out

class CifarLoader(object):
	def __init__(self, source_files):
		self._source = source_files
		self._i = 0
		self.images = None
		self.labels = None 

	def load(self):
		data = [unpickle(f) for f in self._source]
		images = np.vstack([d["data"]for d in data])
		n = len(images)
		self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
		self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
		return self

	def next_batch(self, batch_size):
		x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
		self._i = (self._i+batch_size) % len(self.images)
		return x, y

class CifarDataManager(object):
	def __init__(self):
		self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
		self.test = CifarLoader(["test_batch"]).load()

def display_cifar(images, size):
	n = len(images)
	plt.figure()
	plt.gca().set_axis_off()
	im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) 
		for i in range(size)])
	plt.imshow(im)
	plt.show()


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

# files = os.listdir('C:\\NextGenChatbot\\input_data')
# files = [file for file in files if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file)]

# f = files[0]
# print(plot_image(f))

# img = plt.imread(os.path.join('input_data', f))
# squared_image = imcrop_tosquare(img)
# crop = imcrop(squared_image, 0.2)
# rsz = imresize(crop, (500, 500))
# plt.imshow(rsz)

# img = plt.imread(os.path.join('input_data', f))
# squared_image = imcrop_tosquare(img)
# crop = imcrop(squared_image, 0.1)
# rsz = imresize(img, (500, 500))
# #plt.imshow(rsz)

def copy(old_filename):
    destination_directory = os.path.join("C:\\NextGenChatbot\\input_data_resized")
    source_file = os.path.join("C:\\NextGenChatbot\\input_data", old_filename)
    img = plt.imread(source_file)
    rsz = imresize(img, (500, 500))
    plt.imsave(os.path.join("C:\\NextGenChatbot\\input_data_resized", old_filename), rsz)
    #shutil.copy(source_file, destination_directory)
    
# files = os.listdir("C:\\NextGenChatbot\\input_data")
# for i, filename in enumerate(files):
#     if ('.jpg' in filename) or ('.jpeg' in filename) or ('.png' in filename):
#         print(filename)
#         copy(filename)

def one_hot_encoder(integer_list, unique_labels):
	np_list = np.array(integer_list)
	one_hot_list = np.zeros((len(integer_list), unique_labels))
	one_hot_list[np.arange(len(integer_list)), np_list]=1
	final_list = []
	for list1 in one_hot_list:
		final_list.append(list(map(int, list1)))
	return final_list












