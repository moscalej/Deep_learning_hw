import os
import sys
import cv2
import random
import numpy as np


class DSC:

    T_VALUES = (2, 4, 5)

    def __init__(self, images_path, sizes):
        """

        :param images_path: The path to a directory containing our images
        :param sizes: an iterable of length 3 which represents the number of combinations we would like to make
        for every unique image (within every distinct value of t)
        """
        self.images = self._unpack_images(images_path)
        self.cropped_images = self._shred()
        new_t_2, new_t_4, new_t_5 = self._generate_data(sizes)
        self.new_t_2 = new_t_2
        self.new_t_4 = new_t_4
        self.new_t_5 = new_t_5

    def fit(self, X):
        """
        depending of the t_value value it should shred each sample
        shuffle the picture and get the label

        :param X:
        :return:
        """
        raise NotImplemented

    def _generate_data(self, sizes):
        """

        :param sizes: an iterable of length 3 which represents the number of combinations we would like to make
        for every unique image (within every distinct value of t)
        :return: A new dataset
        """

        new_t_2 = []
        new_t_4 = []
        new_t_5 = []

        for image, crops in self.cropped_images:
            for c in crops[0]:
                for _ in range(sizes[0]):
                    new_t_2.append(self._generate_new_image(c, 2))

            for c in crops[1]:
                for _ in range(sizes[1]):
                    new_t_4.append(self._generate_new_image(c, 4))

            for c in crops[2]:
                for _ in range(sizes[2]):
                    new_t_5.append(self._generate_new_image(c, 5))

        return new_t_2, new_t_4, new_t_5


    @staticmethod
    def _generate_new_image(image, t):
        """

        :param image: a list of crops for a particular image with a particular value t
        :param t: the number of partitions by rows and columns we performed. Either 2, 4, or 5
        :return: A new image composed by reassembling a shuffled set of cropped images
        """

        # Shuffle the crops
        crops = image.copy()
        crops = random.shuffle(crops)

        # Construct a new images from the shuffled crops
        new_img = np.array([])
        for row in range(t):
            new_row = np.array([])
            for column in range(t):
                np.concatenate((new_row, crops[row * t + column]), axis=1)
            np.concatenate((new_img, new_row), axis=0)
        return new_img

    def _unpack_images(self, images_path):
        """

        :param image_path: The path to a certain image
        :return: a list of cv2 image objects in grayscale format
        """
        assert isinstance(images_path, str)
        files = os.listdir(images_path)
        images = filter(lambda x: x[-4:] == ".jpg", files)
        results = []
        for image in images:
            # read image
            im = cv2.imread(os.path.join(images_path, image))
            # convert image to gray scale
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            results.append(im)
        return results

    def _shred(self):
        """
        Shred each image. return in the same order as the initial images
        :return: a dictionary which maps to a list of lists.

                given image index i, we map to 3 lists representing image i. These 3 lists represent:

                    0) the first list is the i'th image cropped with t = 2
                    1) the first list is the i'th image cropped with t = 4
                    2) the first list is the i'th image cropped with t = 5

                we return this outermost dictionary.
        """

        images = {}
        for i in range(len(self.images)):
            im = self.images[i].copy()
            crops = []
            for t in self.T_VALUES:
                t_vals = []
                height = im.shape[0]
                width = im.shape[1]
                frac_h = height // t
                frac_w = width // t
                for h in range(t):
                    for w in range(t):
                        crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                        t_vals.append(crop)
                crops.append(t_vals)
            images[i] = crops
        return images




