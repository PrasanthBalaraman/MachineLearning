from NextGenChatBotDataManager import *
from NextGenChatBotHelper import *
import tensorflow as tf 
import numpy as np 
from helper import *

print("Number of training images: {}".format(len(training_images)))
print("Number of training labels: {}".format(len(training_labels)))
print("Number of test images: {}".format(len(test_images)))
print("Number of test labels: {}".format(len(test_labels)))

# path to save our model
DIR = "./path/to/model"

# initializing the placeholders
x = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
y = tf.placeholder(tf.float32, shape=[None, 3])

# convolutional layer 1
conv1 = conv_layer(x, [5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

# convolutional layer 2
conv2 = conv_layer(conv1_pool, [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

# convolutional layer 3
conv3 = conv_layer(conv2_pool, [5, 5, 64, 64])
conv3_pool = max_pool_2x2(conv3)

# # convolutional layer
# conv4 = conv_layer(conv3_pool, [15, 15, 128, 256])
# conv4_pool = max_pool_2x2(conv4)

# fully connected layer 1 
conv3_flat = tf.reshape(conv3_pool, shape=[-1, 64*64*64])
full1 = full_layer(conv3_flat, 1024)
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

# fully connected layer 2
y_pred = full_layer(full1_drop, 3)

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

# creating collection to export training variables for NextGenChatBot restore module
training_variables =  [y_pred, x, y, accuracy, keep_prob]
tf.add_to_collection('training_variables', training_variables[0])
tf.add_to_collection('training_variables', training_variables[1])
tf.add_to_collection('training_variables', training_variables[2])
tf.add_to_collection('training_variables', training_variables[3])
tf.add_to_collection('training_variables', training_variables[4])

# creating the saver object 
saver = tf.train.Saver(max_to_keep=7, keep_checkpoint_every_n_hours=1)
saver.export_meta_graph(os.path.join(DIR, "NextGenChatBot.meta"), collection_list=['training_variables'])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(number_of_steps):
		i = step 
		x_batch = training_images[i:i+batch_size]
		y_batch = training_one_hot_labels[i:i+batch_size]
		i = (i+batch_size) % len(training_images)

		if step%10==0:
			training_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
			print("Training step {}: accuracy {}".format(step, training_accuracy))
			saver.save(sess, os.path.join(DIR, "NextGenChatBot"), global_step=step)

		sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

	X = test_images
	Y = test_one_hot_labels
	test_accuracy = sess.run(accuracy, feed_dict={x: X, y:Y, keep_prob: 1.0})
	print("Test Accuracy: {}".format(test_accuracy))