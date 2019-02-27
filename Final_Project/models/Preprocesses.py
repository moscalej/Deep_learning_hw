import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from numba import njit
from sklearn.model_selection import train_test_split
from scipy import signal
from random import randint


#  TODO: pre-process for inference (unknown cuts, OOD samples, already shredded)
# def pre_process_data(input_path: str, cuts: int, shape: int = 32) -> list:
def pre_process_data(input_path: list, cuts: int, shape: int = 32, normalize: bool = True) -> list:
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
        x = (x - x_mean) / (x_std + 1e-9)
        print(f' mean {x_mean} , std {x_std}')

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
                image.append([crop_rehaped, i, number_to_angle(i, cuts), neighbours(i, cuts)])
                i = i + 1
        images.append(image)
    # return np.array(images) # todo back to array
    return images


def reshape_all(images: list, sise: int, mean: float, std: float) -> list:
    return list(map(lambda x: (cv2.resize(x, (sise, sise)) - 161.24) / 88.915, images))


# @njit()
def stich(img1, img2, orient):
    # (up, down, left, right)
    if orient == 0:
        additional1 = np.sum(img1, axis=0)
        img1_top = img1[0:2, :]
        additional2 = np.sum(img2, axis=0)
        img2_bottom = img2[-2:, :]
        stiched = np.rot90(np.concatenate([additional2, img2_bottom, img1_top, additional1], axis=0))
    elif orient == 1:
        additional1 = np.sum(img1, axis=0)
        additional2 = np.sum(img2, axis=0)
        img1_bottom = img1[-2:, :]
        img2_top = img2[0:2, :]
        stiched = np.rot90(np.concatenate([additional1, img1_bottom, img2_top, additional2], axis=0))
    elif orient == 2:
        additional1 = np.sum(img1, axis=1)
        additional2 = np.sum(img2, axis=1)
        img1_left = img1[:, 0:2]
        img2_right = img2[:, -2:]
        stiched = np.concatenate([additional2, img2_right, img1_left, additional1], axis=1)
    elif orient == 3:
        additional1 = np.sum(img1, axis=1)
        additional2 = np.sum(img2, axis=1)
        img1_right = img1[:, -2:]
        img2_left = img2[:, 0:2]
        stiched = np.concatenate([additional1, img1_right, img2_left, additional1], axis=1)
    else:
        return "error"
    return stiched


def dissimilarity(edge_a, edge_b):
    # todo see if reshape needed
    return float(np.sum(np.sqrt((edge_a - edge_b) ** 2)))


def choose_false_crops(image, target_crop, options, size):
    # tuple lists (diss, crop_ind)
    diss_top = []
    diss_down = []
    diss_left = []
    diss_right = []
    target_edge_top = target_crop[1, :]
    target_edge_down = target_crop[-1, :]
    target_edge_left = target_crop[:, 1]
    target_edge_right = target_crop[:, -1]
    for crop_ind in options:
        crop = image[crop_ind][0]
        crop_top = crop[1, :]
        crop_down = crop[-1, :]
        crop_left = crop[:, 1]
        crop_right = crop[:, -1]
        diss_top.append((dissimilarity(target_edge_top, crop_top), crop_ind))
        diss_down.append((dissimilarity(target_edge_down, crop_down), crop_ind))
        diss_left.append((dissimilarity(target_edge_left, crop_left), crop_ind))
        diss_right.append((dissimilarity(target_edge_right, crop_right), crop_ind))
    diss_top.sort(key=lambda x: x[0], reverse=True)
    diss_down.sort(key=lambda x: x[0], reverse=True)
    diss_left.sort(key=lambda x: x[0], reverse=True)
    diss_right.sort(key=lambda x: x[0], reverse=True)
    combined_list = [diss_top, diss_down, diss_left, diss_right]
    chosen_crops = []
    top_ind, down_ind, left_ind, right_ind = 0, 0, 0, 0
    combined_inds = [top_ind, down_ind, left_ind, right_ind]
    for i in range(size):
        orient = int(
            np.argmax([diss_top[top_ind][0], diss_down[down_ind][0], diss_left[left_ind][0], diss_right[right_ind][0]]))
        chosen_crop_ind = combined_list[orient][combined_inds[orient]][1]
        chosen_crop = image[chosen_crop_ind][0]
        combined_inds[orient] += 1
        chosen_crops.append([chosen_crop, orient])
    return chosen_crops


# @njit()
def processed2train(images: list, axis_size, mode='train') -> tuple:
    trainX = []
    trainY = []
    for im_ind, image in enumerate(images):
        # OOD_inds = (x for x in np.random.choice([i for i in range(len(images)) if i != im_ind], size=axis_size))
        # crop_with_OOD = np.random.choice(range(len(image)), size=axis_size)
        for crop_ind, crop in enumerate(image):
            neighs = crop[3]  # neighbours(up, down, left, right)
            num_neigh = np.count_nonzero(np.array(neighs) + 1)
            true_crops = [(image[neigh_ind][0], orient) for orient, neigh_ind in enumerate(neighs) if neigh_ind != -1]
            true_crops_inds = [(neigh_ind, orient) for orient, neigh_ind in enumerate(neighs) if neigh_ind != -1]
            neigh_inds = [neigh for (neigh, _) in true_crops_inds if neigh != -1]
            # false_crops_inds = np.random.choice([i for i in range(len(image)) if i not in set(neigh_inds) | {crop_ind}],
            #                                     size=num_neigh)
            if mode == 'train':
                false_crops = choose_false_crops(image, crop[0],
                                                 [i for i in range(len(image)) if
                                                  i not in set(neigh_inds) | {crop_ind}],
                                                 num_neigh)
            elif mode == 'validation':
                false_crops = choose_false_crops(image, crop[0],
                                                 [i for i in range(len(image)) if
                                                  i not in set(neigh_inds) | {crop_ind}],
                                                 len(image) - num_neigh - 1)
            # if crop_ind in crop_with_OOD:
            #     false_crops[0] = images[next(OOD_inds)][int(np.random.choice(range(axis_size ** 2), size=1))]

            # add true samples
            for (true_c, orient) in true_crops:
                trainX.append(stich(crop[0], true_c, orient))
                trainY.append(1)

            # add false samples
            for (false_c, orient_) in false_crops:
                trainX.append(stich(crop[0], false_c, orient_))
                trainY.append(0)

    trainX = np.array(trainX)
    trainY = to_categorical(trainY, 2)
    trainX, _, trainY, _ = train_test_split(trainX, trainY, test_size=0)

    return trainX, trainY


def processed2train_2_chanel(images: list, axis_size) -> tuple:  # todo back to array
    trainX_0 = []
    trainX_1 = []
    trainY = []
    number_of_crops = len(images[0])
    number_images=len(images)
    all_crops = set(range(number_of_crops))
    for im_ind, image in enumerate(images):  # todo back to array
        for crop, crop_index, _, neighs in image:

            true_crops = [(image[neigh_ind][0], orient) for orient, neigh_ind in enumerate(neighs) if neigh_ind != -1]
            true_crops_inds = [(neigh_ind, orient) for orient, neigh_ind in enumerate(neighs) if neigh_ind != -1]
            neigh_inds = [neigh for (neigh, _) in true_crops_inds if neigh != -1]
            list_1 = list(all_crops - (set(neigh_inds) | {crop_index}))

            for true_c, orient in true_crops:
                trainX_0.append(crop)
                trainX_1.append(true_c)
                trainY.append(orient)

            # add false samples

            trainX_0.append(crop)
            trainX_1.append(image[np.random.choice(list_1)][0])
            trainY.append(4)
            for i in range(3):
                trainX_0.append(crop)
                trainX_1.append(images[randint(0,number_images-1)][randint(0,number_of_crops-1)][0])
                trainY.append(4)


    trainX_0 = np.expand_dims(np.array(trainX_0), axis=3)
    trainX_1 = np.expand_dims(np.array(trainX_1), axis=3)
    trainY = to_categorical(trainY, 5)
    trainX_0, _, trainX_, _, trainY, _ = train_test_split(trainX_0, trainX_1, trainY, test_size=0, shuffle=False)
    return trainX_0, trainX_1, trainY


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
    This function will give each picture there neighbors
    :param number: where in the grid
    :type number: int
    :param number_sectors: number of cuts
    :type number_sectors: int
    :return: Who is the neighbors -1 means no one (up, down, left, right)
    :rtype: list
    """
    col = number % number_sectors
    row = number // number_sectors

    neighbors = [number - number_sectors, number + number_sectors, number - 1, number + 1]

    if row == 0:
        neighbors[0] = -1
    if row == number_sectors - 1:
        neighbors[1] = -1
    if col == 0:
        neighbors[2] = -1
    if col == number_sectors - 1:
        neighbors[3] = -1
    return neighbors


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


def plot_preproccess(true_im: list, bighbors: list, orient: list, index: int):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(true_im[index].squeeze())
    ax[1].imshow(bighbors[index].squeeze())

    plt.title(f"the oritation is {['up', 'down', 'left', 'right', 'Not'][int(np.argmax(orient[index]))]}")
    plt.show()


def preprocess_inference(images, shape):
    return None
