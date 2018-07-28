import os
import cv2
import random
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class DSC:
    T_VALUES = (2, 4, 5)

    def __init__(self, images_path, t_value, num_gen):
        """

        :param images_path: The path to a directory containing our images
        :param t_value: the t value we use to partition each image
        :param num_gen: the number of permutations of the crops of a particular image
        """
        # TODO: create a function which determines num gen automatically as a function of t_value
        self.t_value = t_value
        self.num_gen = num_gen
        self.images = self._unpack_images(images_path)
        self.image_crops = self._create_crops()

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
            im = cv2.resize(im, (224, 224))
            results.append(im)
        return results

    def _create_crops(self):
        """

        :return: a dictionary mapping the image index to a dictionary
        of crops (i.e., a crop index to the crop object).
        """
        result = {}
        for i in range(len(self.images)):
            result[i] = self._shred(i)
        return result

    def _shred(self, ind):
        """

        Shred image <ind> into <tval> vertical and horizontal partitions.

        :param ind: the index of the image we would like to shred.
        :param tval: the t value for the shredder. How equidistant vertical/horizontal cuts
        do we make
        :return: a shredded version of the object with the appropriate tval. returned
        in order of initial list of images.
        """

        result = {}
        tval = self.t_value

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
            w = 0
            h += 1
        return result

    def fit(self, X):
        """
        depending of the t_value value it should shred each sample
        shuffle the picture and get the label

        :param X:
        :return:
        """
        raise NotImplemented

    def generate_batch(self, batch_size):
        """

        :param batch_size: the desired batch size
        :param specs:
        :return:
            1) a tensor of dimensions <batch size> x <pictures> x <224> x <224>
            2) a tensor of dimensions <batch size> x
                <Matrix of One hot representation of labels of crops in a particular image>
        """
        image_size = len(self.images)
        place = 0
        index = 0
        while (True):
            image_tensor = np.zeros([batch_size, 224, 224, 1])
            sequence = []
            for index in range(batch_size):
                image, order = self._generate_new_image(index + place)
                image_tensor[index] = image.reshape([224, 224, 1])
                sequence.append(np.array(order))

            yield image_tensor, to_categorical(np.array(sequence))
            place = (place + index) % image_size

    def _generate_data_for_crop(self, crops, num_gen, tval):
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

    def _generate_new_image(self, ind):
        """

        :param ind: the index of the image we would like to return a new image for.
        :return:
            1) A new shuffled image of dimensions 224 x 224
            2) the order of the crops
        """

        # Shuffle the crops
        order = [x for x in range(self.t_value ** 2)]
        random.shuffle(order)
        order_iter = iter(order)
        # Construct a new images from the shuffled crops
        rows = []
        for row in range(self.t_value):
            new_row = np.array([])
            line_list = [self.image_crops[ind][next(order_iter)].copy() for column in range(self.t_value)]
            rows.append(np.hstack(line_list))
        new_img = np.vstack(rows)
        return cv2.resize(new_img, (224, 224)), np.array(order)


if __name__ == "__main__":
    img_path = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\images"
    # shredded_image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\shredded_images"
    dsc = DSC(images_path=img_path, t_value=3, num_gen=4)
    new_imge, order = dsc._generate_new_image(0)
    iter3 = dsc.generate_batch(10)
    a, b = next(iter3)
    plt.imshow(new_imge)
