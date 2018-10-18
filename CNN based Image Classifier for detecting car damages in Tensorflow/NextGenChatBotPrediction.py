from NextGenChatBotDataManager import *
import tensorflow as tf 
import os 
import sys 
from PIL import Image
import numpy as np
from NextGenChatBotHelper import * 

# less tensorflow log settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print the filename
print("Uploaded file name is ", sys.argv[1])
input_image_array = array(Image.open(sys.argv[1]).convert('RGB'))
input_image_array_reshaped = np.reshape(input_image_array, (1, 512, 512, 3))

# resetting the default graph and path where model data is stored
tf.reset_default_graph()
DIR = "./path/to/model"

with tf.Session() as sess:
    # importing the stored graph
    saver = tf.train.import_meta_graph(os.path.join(DIR, "NextGenChatBot.meta"))
    saver.restore(sess, os.path.join(DIR, "NextGenChatBot-90"))

    # importing the training variables 
    y_pred = tf.get_collection('training_variables')[0]
    x = tf.get_collection('training_variables')[1]
    y = tf.get_collection('training_variables')[2]
    accuracy = tf.get_collection('training_variables')[3]
    keep_prob = tf.get_collection('training_variables')[4]

    softmax_pred = tf.nn.softmax(logits=y_pred)

    prediction_output = sess.run(y_pred, feed_dict={x: input_image_array_reshaped, keep_prob: 1.0})[0]
    softmax_prediction = sess.run(softmax_pred, feed_dict={x: input_image_array_reshaped, keep_prob: 1.0})[0]
    print(prediction_output)
    print(softmax_prediction)
