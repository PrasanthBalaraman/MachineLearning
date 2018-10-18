import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

data_directory = "C:\\Deep Learning\\LearningTensorflow\\tmp\\data"
number_of_steps = 1000
minibatch_size = 100

data = input_data.read_data_sets(data_directory, one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32)

y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y_predicted = tf.matmul(x, W)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predicted, labels=y_true))

training_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_mask = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, dtype=tf.float32))

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	for step in range(number_of_steps):
		batch_xs, batch_ys = data.train.next_batch(minibatch_size)
		sess.run(training_step, feed_dict={x: batch_xs, y_true: batch_ys})

	test_accuracy = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(test_accuracy*100))