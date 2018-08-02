import cv2
import numpy as np
import random
import matplotlib.pylab as plt
from keras.utils import to_categorical

from models.Data_set_creator import DSC

class DSCM(DSC):
    def _create_crops(self):
        return super()._create_crops()


    def generate_batch(self, batch_size):
        image_size = len(self.images)
        place = 0
        index = 0
        while True:
            image_tensor = np.zeros([batch_size, self.t_value**2, 96, 96, 1])
            sequence = []
            for index in range(batch_size):
                image, order_r = self._generate_new_image(index + place)
                image_tensor[index] = image.reshape([self.t_value**2, 96, 96, 1])
                sequence.append(np.array(order_r))

            yield image_tensor, to_categorical(np.array(sequence))
            place = (place + index) % (image_size - 2 * batch_size)

    def _generate_new_image(self, ind):

        order_l = [x for x in range(self.t_value ** 2)]
        random.shuffle(order_l)
        images = np.array([self.image_crops[ind][pic] for pic in order_l])

        return images, np.array(order_l)
if __name__ == '__main__':

    img_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\test"
    # shredded_image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\shredded_images"
    dsc = DSCM(images_path=img_path, t_value=5)
    new_imge, order = dsc._generate_new_image(5)
    iter3 = dsc.generate_batch(5)
    a, b = next(iter3)
    plt.imshow(new_imge[2])

    plt.show()

