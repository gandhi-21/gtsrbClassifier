# File to convert images in the training and test dataset to csv files to be used by the model
import os

import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

def convert_images(height, width, channels, classes):

    data = []
    labels = []

    classes = classes

    for i in range(classes):
        path = "input/Images/" + format(i, '05d') + '/'
#        print(path)
        Classes = os.listdir(path)
        for im in Classes:
            try:
                image = cv2.imread(path + im)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                labels.append(i)
            except:
                print("")

    data_np = np.array(data)
    labels_np = np.array(labels)



    labels_np = labels_np.reshape(len(labels_np), 1)
    label_encoder = OneHotEncoder()
    label_encoder.fit(labels_np)
    labels_np = label_encoder.transform(labels_np).toarray()


    s = np.arange(data_np.shape[0])
    np.random.seed(42)
    np.random.shuffle(s)
    data_np = data_np[s]
    labels_np = labels_np[s]


    return data_np, labels_np
