import os
import cv2
from models.DSC_M import DSCM
import numpy as np
import keras
from models.rnn_clasic import benchmark_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def predict(model, image):

    labels = model.predict(image)

    # here comes your code to predict the labels of the images
    return labels


def evaluate(file_dir='example/', model_path=None):
    """

    :param file_dir: the path to a particular image
    :param model_path: the path to a directory which contains models for various t values (2,4,5)
    :return: an array of integers representing the correct order of the proper present in the file_dir folder.
    """

    files = os.listdir(file_dir)
    files.sort()

    results = []
    for index, image in enumerate(files):
        if index % 100 == 0: print(image)
        # read image
        im = cv2.imread(os.path.join(file_dir, image))
        # convert image to gray scale
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (96, 96)).reshape([96, 96, 1])

        results.append(im)

    image = np.array(results)
    t = image.shape[0]

    model = keras.models.load_model(os.path.join(model_path, str(t)))
    Y = model.predict(image)

    return Y


