# make a function to do early stopping
# make a function to plot the loss and the average moving values
# make a function to determine the initial learning rate
# make a function to calulate the log loss

import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME') + b
    return tf.nn.relu(x)

def max_pool2(x, ksize=2, strides=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME')

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)

