import tensorflow as tf 
import numpy as np 
from helper import *
from tensorflow.examples.tutorials.mnist import input_data

# initializing the placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# reshaping the images 
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

# convolutional layer 1
conv1 = conv_layer(x_image, [5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

# convolutional layer 2
conv2 = conv_layer(conv1_pool, [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

# fully connected layer 1 
conv2_flat = tf.reshape(conv2_pool, shape=[-1, 7*7*64])
full1 = full_layer(conv2_flat, 1024)
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

# fully connected layer 2
y_pred = full_layer(full1_drop, 10)

# loading data from MNIST dataset
data_directory = "C:\\Deep Learning\\LearningTensorflow\\tmp\\data" 
data = input_data.read_data_sets(data_directory, one_hot=True)

# cross entropy 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

# Optimizer 
learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# training accuracy 
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# iteration parameters 
number_of_steps = 100
batch_size = 50

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(number_of_steps):
		x_batch, y_batch = data.train.next_batch(batch_size)

		if step%1==0:
			training_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
			print("Training step {}: accuracy {}".format(step, training_accuracy))

		sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

	X = data.test.images.reshape(10, 1000, 784)
	Y = data.test.labels.reshape(10, 1000, 10)
	test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y:Y[i], keep_prob: 1.0}) for i in range(10)]) 
	print("Test Accuracy: {}".format(test_accuracy))














