import tensorflow as tf 
import os 
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

tf.reset_default_graph()
DIR = "./path/to/model"

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(
		os.path.join(DIR, "mnist_checkpoint.meta"))
	saver.restore(sess,
		os.path.join(DIR, "mnist_checkpoint-100"))

	x = tf.get_collection('train_var')[0]
	y_true = tf.get_collection('train_var')[1]
	accuracy = tf.get_collection('train_var')[2]

	ans = sess.run(accuracy, feed_dict=
		{x: data.test.images,
		y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))