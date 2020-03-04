
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

        conv1 = conv2d(self.x, weights["cl1"], biases["bcl1"])
        # conv1_2
        conv1_2 = conv2d(conv1, weights["cl1_2"], biases["bcl1_2"])
        # pool1
        conv1_2 = max_pool2(conv1_2)
        # conv2_1
        conv2 = conv2d(conv1_2, weights["cl2_1"], biases["bcl2_1"])
        # conv2_2
        conv2_2 = conv2d(conv2, weights["cl2_2"], biases["bcl2_2"])
        # pool2
        conv2_2 = max_pool2(conv2_2)
        # conv3_1
        conv3 = conv2d(conv2_2, weights["cl3_1"], biases["bcl3_1"])
        # conv3_2
        conv3_2 = conv2d(conv3, weights["cl3_2"], biases["bcl3_2"])
        # conv3_3
        conv3_3 = conv2d(conv3_2, weights["cl3_3"], biases["bcl3_3"])
        # pool3
        conv3_3 = max_pool2(conv3_3)
        # conv4_1
        conv4 = conv2d(conv3_3, weights["cl4_1"], biases["bcl4_1"])
        # conv4_2
        conv4_2 = conv2d(conv4, weights["cl4_2"], biases["bcl4_2"])
        # conv4_3
        conv4_3 = conv2d(conv4_2, weights["cl4_3"], biases["bcl4_3"])
        # pool4
        conv4_3 = max_pool2(conv4_3)
        # conv5_1
        conv5 = conv2d(conv4_3, weights["cl5_1"], biases["bcl5_1"])
        # conv5_2
        conv5_2 = conv2d(conv5, weights["cl5_2"], biases["bcl5_2"])
        # conv5_3
        conv5_3 = conv2d(conv5_2, weights["cl5_3"], biases["bcl5_3"])
        # pool5
        conv5_3 = max_pool2(conv5_3)
        # fc1
        shape = int(np.prod(conv5_3.get_shape()[1:]))
        fc1_w = tf.Variable(tf.random_normal([shape, 4096]))
        fc1_b = tf.Variable(tf.random_normal([4096]))
        conv5_3_flat = tf.reshape(conv5_3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(conv5_3_flat, fc1_w), fc1_b)
        fc1l = tf.nn.relu(fc1l)
        # fc2
        fc2_w = tf.Variable(tf.random_normal([4096, 4096]))
        fc2_b = tf.Variable(tf.random_normal([4096]))
        fc2l = tf.nn.bias_add(tf.matmul(fc1l, fc2_w), fc2_b)
        fcl2 = tf.nn.relu(fc2l)
        # fc3
        fc3_w = tf.Variable(tf.random_normal([4096, 43]))
        fc3_b = tf.Variable(tf.random_normal([43]))
        fcl3 = tf.nn.bias_add(tf.matmul(fcl2, fc3_w), fc3_b)
        fcl3 = tf.nn.relu(fcl3)
       
        return fcl3


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
        "cl1": tf.Variable(tf.random_normal([3, 3, 3, 64])),
        "cl1_2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
        "cl2_1": tf.Variable(tf.random_normal([3, 3, 64, 128])),
        "cl2_2": tf.Variable(tf.random_normal([3, 3, 128, 128])),
        "cl3_1": tf.Variable(tf.random_normal([3, 3, 128, 256])),
        "cl3_2": tf.Variable(tf.random_normal([3, 3, 256, 256])),
        "cl3_3": tf.Variable(tf.random_normal([3, 3, 256, 256])),
        "cl4_1": tf.Variable(tf.random_normal([3, 3, 256, 512])),
        "cl4_2": tf.Variable(tf.random_normal([3, 3, 512, 512])),
        "cl4_3": tf.Variable(tf.random_normal([3, 3, 512, 512])),
        "cl5_1": tf.Variable(tf.random_normal([3, 3, 512, 512])),
        "cl5_2": tf.Variable(tf.random_normal([3, 3, 512, 512])),
        "cl5_3": tf.Variable(tf.random_normal([3, 3, 512, 512])),
    }
    # biases
    biases = {
        # "bc1": tf.Variable(tf.random_normal([32])),
        # "bc2": tf.Variable(tf.random_normal([64])),
        # "bc3": tf.Variable(tf.random_normal([128])),
        # "bd1": tf.Variable(tf.random_normal([labels_size])),
        # "out": tf.Variable(tf.random_normal([labels_size]))
        "bcl1": tf.Variable(tf.random_normal([64])),
        "bcl1_2": tf.Variable(tf.random_normal([64])),
        "bcl2_1": tf.Variable(tf.random_normal([128])),
        "bcl2_2": tf.Variable(tf.random_normal([128])),
        "bcl3_1": tf.Variable(tf.random_normal([256])),
        "bcl3_2": tf.Variable(tf.random_normal([256])),
        "bcl3_3": tf.Variable(tf.random_normal([256])),
        "bcl4_1": tf.Variable(tf.random_normal([512])),
        "bcl4_2": tf.Variable(tf.random_normal([512])),
        "bcl4_3": tf.Variable(tf.random_normal([512])),
        "bcl5_1": tf.Variable(tf.random_normal([512])),
        "bcl5_2": tf.Variable(tf.random_normal([512])),
        "bcl5_3": tf.Variable(tf.random_normal([512])),
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
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
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
                pass
            epochs += 1
            sess.run(initialize_test_iterator)
            
            xtest, ytest = sess.run(values2)
            #print("test values ", xtest.shape, ytest.shape)
                        # loss = sess.run(loss_op, feed_dict={X: xtest, Y: ytest, keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={X: xtest, Y: ytest, keep_prob: 1.0})
            print(f'epoch {epochs} accuracy = {acc}')
        # Calculate epoch accuracy

    print("Training finished")