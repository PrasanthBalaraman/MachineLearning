import tensorflow as tf 
from NextGenChatBotHelper import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# defining some parameters 
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# storing tensorflow model summaries 
LOG_DIR = "logs/RNN_with_summaries"

# create placeholder for inputs and labels 
x = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name="inputs")
y = tf.placeholder(tf.float32, shape=[None, num_classes], name="labels")

# creating weights and bias terms for RNN
with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))

# This helper function, taken from the official TensorFlow documentation, 
# # simply adds some ops that take care of logging summaries
def variable_summaries(var):
	with tf.name_scope("summaries"):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope("stddev"):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar("stddev", stddev)
		tf.summary.scalar("max", tf.reduce_max(var))
		tf.summary.scalar("min", tf.reduce_min(var))
		tf.summary.histogram("histogram", var)

def rnn_step(previous_hidden_state, x):
	current_hidden_state = tf.tanh(
		tf.matmul(previous_hidden_state, Wh) +
		tf.matmul(x, Wx) + 
		b_rnn
	)
	return current_hidden_state

# processing inputs to work with scan function
# current input shape = (batch size, time steps, element size)
processed_input = tf.transpose(x, perm=[1, 0, 2])
# current input shape now = (time steps, batch size, element size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# getting all state vectors across time 
all_hidden_states = tf.scan(rnn_step, processed_input, initializer=initial_hidden, name="states")

# weights for output layers 
with tf.name_scope("linear_layer_weights") as scope:
	with tf.name_scope("W_linear"):
		Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))
		variable_summaries(Wl)
	with tf.name_scope("Bias_linear"):
		bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
		variable_summaries(bl)
	
def get_linear_layer(hidden_state):
	return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope("linear_layer_weights") as scope:
	# iterate across time, apply linear layer to all RNN outputs
	all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
	# get lasrt output 
	output = all_outputs[-1]
	tf.summary.histogram("outputs", output)

with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
	tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
	accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
	tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

# get a small test dataset
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))

test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
	# write summaries to LOG_DIR -- used by TensorBoard
	train_writer = tf.summary.FileWriter(LOG_DIR + "/train", graph=tf.get_default_graph())
	test_writer = tf.summary.FileWriter(LOG_DIR + "/test", graph=tf.get_default_graph())

	sess.run(tf.global_variables_initializer())

	for i in range(10000):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size, time_steps, element_size))
		summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y: batch_y})

		# add to summaries 
		train_writer.add_summary(summary, i)

		if i%1000==0:
			acc, loss, = sess.run([accuracy, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			print("Iter " + str(i) + ", Minibatch Loss= " + \
			"{:.6f}".format(loss) + ", Training accuracy= " + \
			"{:.5f}".format(acc))

		if i%10:
			# calculate accuracy for 128 MNIST test images and add summaries 
			summary, acc = sess.run([merged, accuracy], feed_dict={x: test_data, y: test_label})
			test_writer.add_summary(summary, i)

	test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})

	print("Test Accuracy: ", test_acc)
