# import tensorflow as tf 

# tf.reset_default_graph()

# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])
# v3 = tf.get_variable("v3", shape=[7])

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/tmp/model.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())
#   print("v3 : %s" % v3.eval())

import tensorflow as tf 

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph("test_saver.meta")
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	print(sess.run('v1:0'))
	print(sess.run('v2:0'))
	print(sess.run('v3:0'))