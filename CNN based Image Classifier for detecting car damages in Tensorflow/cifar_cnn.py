import tensorflow as tf 
from NextGenChatBotHelper import *
import numpy as np 

data = CifarDataManager()
print("Number of train images: {}".format(len(data.train.images)))
print("Number of train labels: {}".format(len(data.train.labels)))
print("Number of test images: {}".format(len(data.test.images)))
print("Number of test labels: {}".format(len(data.test.labels)))

# placeholder for input and output
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

# convolutional layer one
initial_1 = tf.truncated_normal([5, 5, 3, 1], stddev=0.1) 
W_1 = tf.Variable(initial_1, dtype=tf.float32)
b_1 = tf.Variable(tf.constant(0.1, shape=[1]))
conv_1 = tf.nn.relu(tf.nn.conv2d(x, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)
conv_1_pool = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# convolutional layer two 
initial_2 = tf.truncated_normal([5, 5, 32, 64], stddev=0.1)
W_2 = tf.Variable(initial_1, dtype=tf.float32)
b_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1_pool, W_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)
conv_2_pool = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# fully connected layer 1 
conv_2_flat = conv_2_pool.reshape(None, 8*8*64)
initial_3 = tf.truncated_normal([8*8*64, 1024], stddev=0.1)
W_3 = tf.Variable(initial_3, dtype=tf.float32)
b_3 = tf.Variable(tf.constant(0.1, shape=[1024]))
full_1 = (tf.matmul(conv_2_flat, W_3) + b_3)
keep_prob = tf.placeholder(tf.float32)
full_1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

# final fully connected layer
initial_4 = tf.truncated([1024, 10], stddev=0.1)
W_4 = tf.Variable(initial_4, dtype=tf.float32) 
b_4 = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32)
y_pred = tf.matmul(full_1_drop, W_4) + b_4

# defining loss 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

# accuracy 
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer 
learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate)
training_step = optimizer.minimize(loss)

# iteration hyperparameters 
number_of_steps = 10
batch_size = 50

DIR = "C:\\NextGenChatbot\\"
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(number_of_steps):
		x_batch, y_batch = data.train.next_batch(batch_size)

		if step%10==0:
			training_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
			print("Training step {}: accuracy {}".format(step, training_accuracy))
			saver.save(sess, os.path.join(DIR, "model"), global_step=step)

		sess.run(training_step, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

	X = data.test.images.reshape(10, 1000, 1024)
	Y = data.test.labels.reshape(10, 1000, 10)
	test_accuracy = sess.run(accuracy, feed_dict={x: X, y:Y, keep_prob: 1.0})
	print("Test Accuracy: {}".format(test_accuracy))



