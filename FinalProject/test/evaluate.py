import os
import cv2
import numpy as np
import keras


def predict(model, image):

    labels = model.predict(image)

    # here comes your code to predict the labels of the images
    return labels


def evaluate(file_dir='example/', model_paths={}):
    """
    :param file_dir: the path to a particular image
    :param model_path: a dictionary which maps from a t value to the model corresponding to that t value
    """

    assert isinstance(model_paths, dict), "We need an inputted dictionary to map from a t_value to a model path"
    assert len(model_paths) == 3, "This dictionary should contain 3 entries"

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

    model = keras.models.load_model(model_paths[t])
    Y = model.predict(image.reshape([1,t,96,96,1]))

    return Y