import tensorflow as tf 

v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
v3 = tf.get_variable("v3", shape=[7], initializer=tf.zeros_initializer)

v1 = v1.assign(v1+1)
v2 = v2.assign(v2-1)
v3 = v3.assign(v3+2)

saver = tf.train.Saver()

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

saver.save(sess, './test_saver')

# import tensorflow as tf
# w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
# w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, './my_test_model')
 
# # This will save following files in Tensorflow v >= 0.11
# # my_test_model.data-00000-of-00001
# # my_test_model.index
# # my_test_model.meta
# # checkpoint

