import tensorflow as tf 
import numpy as np 

x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.7, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000)*0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

number_of_steps = 10

g = tf.Graph()
wb = []

with g.as_default():
	x = tf.placeholder(tf.float32, shape=[None, 3])
	y_true = tf.placeholder(tf.float32, shape=None)

	with tf.name_scope("inference") as scope:
		w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
		b = tf.Variable(0, dtype=tf.float32, name='bias')
		y_pred = tf.matmul(w, tf.transpose(x)) + b

	with tf.name_scope("loss") as scope:
		loss = tf.reduce_mean(tf.square(y_true-y_pred))

	with tf.name_scope("optimize") as scope:
		learning_rate = 0.5
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for step in range(number_of_steps):
			sess.run(train, feed_dict={x: x_data, y_true: y_data})
			print(sess.run([w, b]))
