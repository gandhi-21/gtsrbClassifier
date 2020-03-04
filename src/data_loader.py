# Read in the data and the labels and then shuffle them together or combine and shuffle

import tensorflow as tf
import numpy as np
import pandas as pd

from img_csv import convert_images


class batch_iterator():


    def __init__(self, x_train, y_train, x_test, y_test, epochs, batch_size, random_seed):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

    def pre_process(self, x, y):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.math.divide(x, tf.Variable(255.0))
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        return x, y

    def train_iterator(self):
        print("making a iterator")

        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(len(self.x_train))
        dataset = dataset.map(self.pre_process, num_parallel_calls=4)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator_initialize = iterator.make_initializer(dataset)
        return iterator, iterator_initialize

    def test_iterator(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        dataset = dataset.shuffle(len(self.x_test))
        dataset = dataset.map(self.pre_process, num_parallel_calls=4)
        dataset = dataset.batch(128)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator_initialize = iterator.make_initializer(dataset)
        return iterator, iterator_initialize


# if __name__ == "__main__":

#     x_values, y_values = convert_images(32, 32, 3, 43)

#     training_pipeline = batch_iterator(x_values, y_values, None, None, 10, 128, 42)
#     # print(training_pipeline)
#     iterator = training_pipeline.train_iterator()
#     values = iterator.get_next()
#     sum_x = 0
#     sum_y = 0

#     with tf.Session() as sess:
#         try:
#             while True:
#                 x_batch, y_batch = sess.run(values)
#                 sum_y += len(y_batch)
#                 sum_x += len(x_batch)
#         except tf.errors.OutOfRangeError:
#             pass

#     print(sum_x)
#     print(sum_y)
