#!/usr/bin/env python

#imports
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()

#Bookeeping:
model_path = "model/model.ckpt-201"
checkpath = "ckpt/tfcnn-16000"
train_iter = 2000

#Input Initialization
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
accs = []


#Model Details
#CNN

#Initialization functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution Functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#1st Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Layer 2 (64 features)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Fully Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Checking the Output
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#Doing the work
#sess.run(tf.global_variables_initializer())
#creates the saver
#saver = tf.train.Saver()
#[W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]


#saver = tf.train.Saver()
#init = tf.initialize_all_variables()

#Start the Session
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  # Restore variables from disk.
  rec_saver = tf.train.import_meta_graph(model_path + '.meta')
  rec_saver.restore(sess, model_path)
  print("Model restored.")
  
  for i in range(100):
    batch = mnist.test.next_batch(50)
    accs.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
  
  print('test accuracy %g' % np.mean(accs))

