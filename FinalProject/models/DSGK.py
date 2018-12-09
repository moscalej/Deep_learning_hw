import os
import cv2
import random
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class DSGk:
    T_VALUES = (2, 4, 5)

    def __init__(self, images_path, t_value, images=None):
        """

        :param images:
        :param images_path: The path to a directory containing our images
        :param t_value: the t value we use to partition each image
        """

        self.t_value = t_value
        self.scale = MinMaxScaler((-1,1))
        self.images = images if images is not None else self._unpack_images(images_path)
        self.images = (self.images * (2/255) -1).reshape([self.images.shape[0],480,480,1])
        self.generator = ImageDataGenerator(featurewise_center=False,
                          samplewise_center=False,
                          featurewise_std_normalization=False,
                          samplewise_std_normalization=False,
                          zca_whitening=False,
                          zca_epsilon=1e-06,
                          rotation_range=0.1,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          brightness_range=None,
                          shear_range=0.0,
                          zoom_range=0.1,
                          channel_shift_range=0.0,
                          fill_mode='nearest',
                          cval=0.0,
                          horizontal_flip=True,
                          vertical_flip=True,
                          rescale=None,
                          preprocessing_function=None,
                          data_format=None,
                          validation_split=0.0)
        self.generator.fit(self.images)


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
        for index ,image in enumerate(images):
            if index % 100==0 :print(image)
            # read image
            im = cv2.imread(os.path.join(images_path, image))
            # convert image to gray scale
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, (480, 480))

            results.append(im)
        return np.array(results)

    def _create_crops(self):
        """

        :return: a dictionary mapping the image index to a dictionary
        of crops (i.e., a crop index to the crop object).
        """
        result = {}
        for i in range(len(self.images)):
            result[i] = self._shred(i)
        return result

    def _shred(self, image):
        """

        Shred image <ind> into <tval> vertical and horizontal partitions.

        :param image: the index of the image we would like to shred.
        do we make
        :return: a shredded version of the object with the appropriate tval. returned
        in order of initial list of images.
        """

        result = {}
        tval = self.t_value
        im = image.copy()
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // tval
        frac_w = width // tval

        h = 0
        w = 0
        image = 0

        while h < tval:
            while w < tval:
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                result[image] = cv2.resize(crop, (96, 96))
                image += 1
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

        batch_iter = self.generator.flow(self.images,batch_size = batch_size, shuffle=True)
        while True:
                image_bactch = next(batch_iter)
                sequence = []
                images = []
                for image in image_bactch:
                    image_c, order_r = self._generate_new_image(image)
                    images.append(image_c)
                    sequence.append(np.array(order_r))
                image_tensor = np.array(images)
                image_tensor =image_tensor.reshape([image_tensor.shape[0],self.t_value**2,96,96,1])

                yield image_tensor, to_categorical(np.array(sequence))


    def _generate_new_image(self, image):
        """

        :param ind: the index of the image we would like to return a new image for.
        :return:
            1) A new shuffled image of dimensions 224 x 224
            2) the order of the crops
        """

        # Shuffle the crops
        order_l = [x for x in range(self.t_value ** 2)]
        random.shuffle(order_l)
        img = self._shred(image)
        # Construct a new images from the shuffled crops
        random.shuffle(order_l)
        images = np.array([img[pic] for pic in order_l])

        return images , order_l


if __name__ == "__main__":
    img_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\test"
    # shredded_image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\shredded_images"
    dsc = DSGk(images_path=img_path, t_value=5)
    iter3 = dsc.generate_batch(10)
    a, b = next(iter3)
