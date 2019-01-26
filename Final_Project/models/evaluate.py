import os
import cv2
from Final_Project.models.Algorithm_sketch import predict_2


def predict(images):
    labels = []

    labels = predict_2(images)

    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    return Y
