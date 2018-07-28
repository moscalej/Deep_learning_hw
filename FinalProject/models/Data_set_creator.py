import os
import cv2
import random
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class DSC:
    T_VALUES = (2, 4, 5)

    def __init__(self, images_path, t_value):
        """

        :param images_path: The path to a directory containing our images
        :param t_value: the t value we use to partition each image
        """

        self.t_value = t_value
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

    def generate_batch(self, batch_size):
        """

        :param batch_size: the desired batch size
        :return:
            1) a tensor of dimensions <batch size> x <pictures> x <224> x <224>
            2) a tensor of dimensions <batch size> x
                <Matrix of One hot representation of labels of crops in a particular image>
        """
        image_size = len(self.images)
        place = 0
        index = 0
        while True:
            image_tensor = np.zeros([batch_size, 224, 224, 1])
            sequence = []
            for index in range(batch_size):
                image, order_r = self._generate_new_image(index + place)
                image_tensor[index] = image.reshape([224, 224, 1])
                sequence.append(np.array(order_r))

            yield image_tensor, to_categorical(np.array(sequence))
            place = (place + index) % image_size

    def _generate_new_image(self, ind):
        """

        :param ind: the index of the image we would like to return a new image for.
        :return:
            1) A new shuffled image of dimensions 224 x 224
            2) the order of the crops
        """

        # Shuffle the crops
        order_l = [x for x in range(self.t_value ** 2)]
        random.shuffle(order_l)
        order_iter = iter(order_l)
        # Construct a new images from the shuffled crops
        rows = []
        for row in range(self.t_value):
            line_list = [self.image_crops[ind][next(order_iter)].copy() for _ in range(self.t_value)]
            rows.append(np.hstack(line_list))
        new_img = np.vstack(rows)
        return cv2.resize(new_img, (224, 224)), np.array(order_l)


if __name__ == "__main__":
    img_path = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\images"
    # shredded_image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\shredded_images"
    dsc = DSC(images_path=img_path, t_value=3)
    new_imge, order = dsc._generate_new_image(0)
    iter3 = dsc.generate_batch(10)
    a, b = next(iter3)
    plt.imshow(new_imge)
