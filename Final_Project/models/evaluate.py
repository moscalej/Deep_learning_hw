import os
import cv2
import yaml
from Final_Project.models.Preprocesses import reshape_all


def predict(images):
    labels = []
    images_crop = reshape_all(images, 40)

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
