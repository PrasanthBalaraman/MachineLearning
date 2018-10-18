import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "/tmp/data"
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 100
MINIBATCH_SIZE = 100

DIR = "./path/to/model"

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_pred = tf.matmul(x, W)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
training_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
diff = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(diff, tf.float32))

train_var = [x, y_true, accuracy]
tf.add_to_collection('train_var', train_var[0])
tf.add_to_collection('train_var', train_var[1])
tf.add_to_collection('train_var', train_var[2])

saver = tf.train.Saver(max_to_keep=7, 
	keep_checkpoint_every_n_hours=1)
saver.export_meta_graph(os.path.join(DIR, "mnist_checkpoint.meta"),
	collection_list=['train_var'])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(1, NUM_STEPS+1):
		batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
		sess.run(training_step, feed_dict=
			{x: batch_xs, y_true: batch_ys})

		if step%50==0:
			saver.save(sess, os.path.join(DIR, "mnist_checkpoint"), 
				global_step=step)

	ans = sess.run(accuracy, feed_dict={x: data.test.images,
		y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
