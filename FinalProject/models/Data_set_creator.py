import os
import sys
import cv2
import random
import numpy as np


class DSC:

    T_VALUES = (2, 4, 5)

    def __init__(self, images_path):
        """

        :param images_path: The path to a directory containing our images
        """
        self.images = self._unpack_images(images_path)

    def fit(self, X):
        """
        depending of the t_value value it should shred each sample
        shuffle the picture and get the label

        :param X:
        :return:
        """
        raise NotImplemented

    def generate_data_for_crop(self, crops, num_gen, tval):
        """

        :param crops: a list of components of a single image. An array.
        :param num_gen: The number of images to generate from a particular crop
        :param tval: the t value associated with the crops of this image.
        :return: <num_gen> randomly geneated images (which are 2d numpy arrays) based
        off of the initial crops of a particular images.
        """

        i = 0
        while i < num_gen:
            yield self._generate_new_image(crops, tval)
            i += 1


    @staticmethod
    def _generate_new_image(image, t):
        """

        :param image: a list of crops for a particular image with a particular value t
        :param t: the number of partitions by rows and columns we performed. Either 2, 4, or 5
        :return: A new image composed by reassembling a shuffled set of cropped images
        """

        # Shuffle the crops
        crops = image.copy()
        order = [x for x in range(t**2)]
        random.shuffle(order)

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
        images = []
        for file in files:
            if ".jp" in file.lower():
                images.append(file)

        results = []
        for image in images:
            # read image
            im = cv2.imread(os.path.join(images_path, image))
            # convert image to gray scale
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, (224,224))
            results.append(im)
        return results

    def shred(self, ind, tval):
        """

        Shred image <ind> into <tval> vertical and horizontal partitions.

        :param ind: the index of the image we would like to shred.
        :param tval: the t value for the shredder. How equidistant vertical/horizontal cuts
        do we make
        :return: a shredded version of the object with the appropriate tval. returned
        in order of initial list of images.
        """

        result = {}

        im = self.images[ind].copy()
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // tval
        frac_w = width // tval

        h = 0
        w = 0
        ind = 0

        while h < tval:
            while w < tval:
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                result[ind] = crop
                ind += 1
                w += 1
            h += 1
        return result


if __name__ == "__main__":
    img_path = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\documents"
    shredded_image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\shredded_images"

    dsc = DSC(img_path)



