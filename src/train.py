
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from img_csv import convert_images
from data_loader import batch_iterator
from utils import conv2d, max_pool2, dropout

class model():

    def __init__(self, x, weights, biases, dropout):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.dropout = dropout

    def model_1(self):

        self.x = tf.reshape(self.x, shape=[-1, 32, 32, 3])

        # make all the layers
        # # conv1
        # conv1 = conv2d(self.x, self.weights["wc1"], self.biases["bc1"])
        # print("shape after conv1 ", conv1.shape)
        # # conv2
        # conv2 = conv2d(conv1, self.weights["wc2"], self.biases["bc2"])
        # print("shape after conv2 ", conv2.shape)
        # # max_pool1
        # conv2 = max_pool2(conv2, ksize=2)
        # print("shape after max pool 1 ", conv2.shape)
        # # dropout
        # conv2 = dropout(conv2, self.dropout)
        # print("shape after drop 1 ", conv2.shape)
        # # conv3
        # conv3 = conv2d(conv2, self.weights["wc3"], self.biases["bc3"])
        # print("shape after conv3 ", conv3.shape)
        # # max_pool2
        # conv3 = max_pool2(conv3, ksize=2)
        # print("shape after max pool 2 ", conv3.shape)
        # # dropout
        # conv3 = dropout(conv3, self.dropout)
        # print("shape after drop 2 ", conv3.shape)
        # # flatten
        # conv3 = tf.contrib.layers.flatten(conv3)
        # print("shape after flatten 1 ", conv3.shape)
        # # dense
        # fc1 = tf.add(tf.matmul(conv3, self.weights["wd1"]), self.biases["bd1"])
        # # dense
        # out = tf.add(tf.matmul(fc1, self.weights["out"]), self.biases["out"])

        # conv1 = conv2d(self.x, weights["wc1"], biases["bc1"])
        # conv1 = max_pool2(conv1, ksize=2)

        # conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
        # conv2 = max_pool2(conv2, ksize=2)

        # conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])

        # conv3_shape = conv3.get_shape().as_list()

        # # fc1 = tf.layers.flatten(conv2)
        # fc1 = tf.reshape(conv2, [-1, conv3_shape[1] * conv3_shape[2]])
        # print("fc1 shape ", fc1.shape)
        # #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        # fc1 = tf.nn.relu(fc1)
        # # print("fc1 shape", fc1.shape)
        
        # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        # # print(out.shape)

        conv1 = conv2d(self.x, weights['cl1'], biases['bl1'])
        conv2 = conv2d(conv1, weights['cl2'], biases['bl2'])
        max_1 = max_pool2(conv2)
        dropout1 = dropout(max_1, keep_prob=keep_prob)
        conv3 = conv2d(dropout1, weights['cl3'], biases['bl3'])
        max_2 = max_pool2(conv3)
        dropout2 = dropout(max_2, keep_prob=keep_prob)
        layer_shape = dropout2.get_shape()
        num_features = layer_shape[1:4].num_elements()
        flatten = tf.reshape(dropout2, [-1, num_features])
        dense = tf.matmul(flatten, weights['d1']) + biases['d1']
        dropout3 = dropout(dense, keep_prob=keep_prob)
        dense2 = tf.matmul(dropout3, weights['d2']) + biases['d2']
        return dense2


if __name__ == "__main__":
    print("Training the model")
    
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    BATCH_SIZE = 128
    image_size = 32
    labels_size = 43
    DROPOUT = 0.25

    # placeholders
    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, shape=[None, labels_size])
    keep_prob = tf.placeholder(tf.float32)

    # weights
    weights = {
        # "wc1": tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # "wc2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # "wc3": tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # "wd1": tf.Variable(tf.random_normal([64, labels_size])),
        # "out": tf.Variable(tf.random_normal([labels_size, labels_size]))
        "cl1": tf.Variable(tf.random_normal([5, 5, 3, 32])),
        "cl2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
        "cl3": tf.Variable(tf.random_normal([5, 5, 64, 128])),
        "d1": tf.Variable(tf.random_normal([8192, 128])),
        "d2": tf.Variable(tf.random_normal([128, labels_size]))
    }
    # biases
    biases = {
        # "bc1": tf.Variable(tf.random_normal([32])),
        # "bc2": tf.Variable(tf.random_normal([64])),
        # "bc3": tf.Variable(tf.random_normal([128])),
        # "bd1": tf.Variable(tf.random_normal([labels_size])),
        # "out": tf.Variable(tf.random_normal([labels_size]))
        "bl1": tf.Variable(tf.random_normal([32])),
        "bl2": tf.Variable(tf.random_normal([64])),
        "bl3": tf.Variable(tf.random_normal([128])),
        "d1": tf.Variable(tf.random_normal([128])),
        "d2": tf.Variable(tf.random_normal([labels_size]))
    }

 # Getting the data
    x, y = convert_images(32, 32, 3, 43)
    # actual data values
    x_values, x_test, y_values, y_test = train_test_split(x, y, test_size=0.2)

   # print(x_values[0].shape)
    print(x_test.shape)
    print(y_test.shape)

    training_data = batch_iterator(x_values, y_values, x_test, y_test, EPOCHS, BATCH_SIZE, 42)
    iterator, initialize_iterator = training_data.train_iterator()
    values = iterator.get_next()

    test_iterator, initialize_test_iterator = training_data.test_iterator()
    values2 = test_iterator.get_next()

    Model = model(X, weights, biases, keep_prob)
    logits = Model.model_1()
    logits = tf.reshape(logits, [-1, labels_size])
    print("logits shape ", logits.shape)
    predictions = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # initialize the variables
    init = tf.global_variables_initializer()
    # set gpu options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # start training

    with tf.Session(config=config) as sess:

        sess.run(init)
        sess.run(initialize_iterator)

        epochs = 0

        for epoch in range(EPOCHS):
            try:
                print(f"training epoch {epochs}")
                while True:
                    batch_x, batch_y = sess.run(values)
                    #print("batch shapes ", batch_x.shape, batch_y.shape)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y,keep_prob: 0.25})
            except tf.errors.OutOfRangeError:
                epochs += 1
                sess.run(initialize_iterator)
                sess.run(initialize_test_iterator)
                try:
                    while True:
                        xtest, ytest = sess.run(values2)
                        print("test values ", xtest.shape, ytest.shape)
                        # loss = sess.run(loss_op, feed_dict={X: xtest, Y: ytest, keep_prob: 1.0})
                        acc = sess.run(accuracy, feed_dict={X: xtest, Y: ytest, keep_prob: 1.0})
                        print(f'epoch {epochs} accuracy = {acc}')
                except tf.errors.OutOfRangeError:
                    pass
                pass
        # Calculate epoch accuracy

    print("Training finished")