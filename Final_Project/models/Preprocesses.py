import cv2
import numpy as np
import os
from numba import njit
from sklearn.model_selection import train_test_split
from scipy import signal


#  TODO: pre-process for inference (unknown cuts, OOD samples, already shredded)
# def pre_process_data(input_path: str, cuts: int, shape: int = 32) -> np.ndarray:
def pre_process_data(input_path: str, cuts: int, shape: int = 32, normalize: bool = True) -> np.ndarray:
    """
    this function will pre process the data making a list of touples where
     it will return an 3d array [image,section,tag]

    :param normalize:
    :type normalize:
    :param shape: Shape of the picture segment resize
    :type shape: int

    :param input_path: path of the folder where the pictures are store
    :type input_path: str list
    :param cuts: #
    :return:
    """
    images = []
    images_uncut = []
    for files_path in input_path:

        files = os.listdir(files_path)  # TODO paths
        for f in files:
            file_path = f'{files_path}/{f}'
            im_uncut = cv2.imread(file_path)
            im_uncut = cv2.cvtColor(im_uncut, cv2.COLOR_RGB2GRAY)
            images_uncut.append(cv2.resize(im_uncut, (shape * cuts, shape * cuts)))
    x = np.array(images_uncut)

    if normalize:
        x_mean = np.mean(x, axis=(0, 1, 2))
        x_std = np.std(x, axis=(0, 1, 2))
        x_center = (x - x_mean) / (x_std + 1e-9)

    for im in x:
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // cuts
        frac_w = width // cuts
        i = 0
        image = []
        for h in range(cuts):
            for w in range(cuts):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                crop_rehaped = cv2.resize(crop, (shape, shape))
                i = i + 1
                image.append([crop_rehaped, i, number_to_angle(i, cuts), niegbors(i, cuts)])
        images.append(image)
    return np.array(images)


@njit()
def stich(img1, img2, orient):
    # (up, down, left, right)
    if orient == 0:
        stiched = np.rot90(np.concatenate((img2, img1), axis=0))
    if orient == 1:
        stiched = np.rot90(np.concatenate((img1, img2), axis=0))
    if orient == 2:
        stiched = np.concatenate((img2, img1), axis=1)
    if orient == 3:
        stiched = np.concatenate((img1, img2), axis=1)
    else:
        return "error"
    sobelx = cv2.Sobel(stiched, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(stiched, cv2.CV_64F, 0, 1, ksize=5)
    return np.stack([sobelx, sobely, stiched], 0)
    # return stiched


# @njit()
def processed2train(images: np.ndarray, axis_size) -> list:
    trainX = []
    trainY = []
    for im_ind, image in enumerate(images):
        OOD_inds = (np.random.choice([i for i in range(len(images)) if i != im_ind], size=axis_size))
        crop_with_OOD = np.random.choice(range(len(image)), size=axis_size)
        for crop_ind, crop in enumerate(image):
            neighs = crop[3]  # neighbours(up, down, left, right)
            num_neigh = np.count_nonzero(np.array(neighs) + 1)
            true_crops = [(neigh_ind, orient) for orient, neigh_ind in neighs if neigh_ind != -1]
            neigh_inds = [neigh for (neigh, _) in true_crops if neigh != -1]
            false_crops = np.random.choice([i for i in range(len(image)) if i not in neigh_inds + [crop_ind]],
                                           size=num_neigh)
            if crop_ind in crop_with_OOD:
                false_crops[0] = next(OOD_inds)

            # add true samples
            for (true_c, orient) in true_crops:
                trainX.append(stich(crop[0], image[true_c], orient))
                trainY.append(1)

            # add false samples
            for (false_c, orient_) in false_crops:
                trainX.append(stich(crop[0], image[false_c][0], np.random.choice([0, 1, 2, 3], size=1)))
                trainY.append(0)

    return trainX, trainY


# @njit()
def create_train_set(input_paths, shape=32):
    trainX = []
    trainY = []
    for input_path in input_paths:
        for cut_size in [2, 4, 5]:
            processed = pre_process_data(input_path, cut_size, shape)
            X, Y = processed2train(processed, cut_size)
            trainX.extend(X)
            trainY.extend(Y)

    return trainX, trainY


@njit()
def neighbours(number: int, number_sectors: int) -> [int, int, int, int]:
    """
    This function will give each picture there neigbors
    :param number: where in the grid
    :type number: int
    :param number_sectors: number of cuts
    :type number_sectors: int
    :return: Who is the nigbors -1 means no one (up, down, left, right)
    :rtype: list
    """
    col = number % number_sectors
    row = number // number_sectors
    nieg = [row - 1, row + 1, col - 1, col + 1]
    if row == number_sectors:
        nieg[1] = -1
    if col == number_sectors:
        nieg[3] = -1
    return nieg


@njit()
def asCartesian(rthetaphi: [np.float, np.float, np.float]) -> [np.float, np.float, np.float]:
    """

    :param rthetaphi: [np.float, np.float, np.float]
    :type rthetaphi:
    :return:
    :rtype:
    """
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1] * np.pi / 180  # to radian
    phi = rthetaphi[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]


@njit()
def number_to_angle(number: int, number_sectors: int) -> [np.float, np.float, np.float]:
    """
    Gives cordinates for a picture given sector
    :param number:
    :param number_sectors:
    :return:
    """
    angles_phi = np.linspace(-22.5, 22.5, number_sectors)
    angles_theta = np.linspace(67.5, 112.5, number_sectors)
    theta = number // number_sectors
    phi = number % number_sectors
    return asCartesian([1, angles_theta[theta], angles_phi[phi]])


#
# @njit()
# def angle_to_number(angle,number_sectors):
#     """
#     need to define
#     :param angle:
#     :param number_sectors:
#     :return:
#     """
#     pass
#     return 0

def for_embeding(data, normalize=True):
    data_m = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    x = np.array(list(map(np.array, data_m[:, 0])))
    y = np.array(list(map(np.array, data_m[:, 2])))
    if normalize:
        x_mean = np.mean(x, axis=(0, 1, 2))
        x_std = np.std(x, axis=(0, 1, 2))
        x_center = (x - x_mean) / (x_std + 1e-9)
    else:
        x_center = x
    x_center = np.expand_dims(x_center, axis=3)
    x_center, _, y, _ = train_test_split(x_center, y, test_size=0)
    return x_center, y

    # isinstance(data,np.ndarray)
