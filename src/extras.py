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