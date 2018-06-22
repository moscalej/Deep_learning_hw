import os, sys
import cv2

Xa = []
Xb = []
y = []

IM_DIR = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\documents"
OUTPUT_DIR = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\output\5x5"
files = os.listdir(IM_DIR)

# update this number for 4X4 crop 2X2 or 5X5 crops.
tiles_per_dim = 5

for f in files:
    im = cv2.imread(f'{IM_DIR}\\{f}')
    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    height = im.shape[0]
    width = im.shape[1]
    frac_h = height//tiles_per_dim
    frac_w = width//tiles_per_dim
    i=0
    for h in range(tiles_per_dim):
        for w in range(tiles_per_dim):

            crop = im[h*frac_h:(h+1)*frac_h,w*frac_w:(w+1)*frac_w]
            cv2.imwrite(f'{OUTPUT_DIR}\\{f[:-4]}_{i}.jpg', crop)
            i=i+1

class DSC:

    T_VALUES = (2, 4, 5)

    def __init__(self, t_value, images_path):
        """

        """
        self.images = self._unpack_images(images_path)
        self.cropped_images = self._shred()

    def fit(self, X):
        """
        depending of the t_value value it should shred each sample
        shuffle the picture and get the label

        :param X:
        :return:
        """

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




