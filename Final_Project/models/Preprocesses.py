"""
# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors. Title to the Material remains with Intel Corp-
# oration or its suppliers and licensors. The Material contains trade
# secrets and proprietary and confidential information of Intel Corpor-
# ation or its suppliers and licensors. The Material is protected by world-
# wide copyright and trade secret laws and treaty provisions. No part of
# the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellect-
# ual property right is granted to or conferred upon you by disclosure or
# delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property
# rights must be express and approved by Intel in writing.
#
#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#           Michal Schachter <michal.schachter@intel.com>
#
"""
import cv2
import numpy as np
import os
from numba import njit
from sklearn.model_selection import train_test_split


def pre_process_data(input_path: str, cuts: int, shape: int = 32) -> np.ndarray:
    """
    this function will pre process the data making a list of touples where
     it will return an 3d array [image,section,tag]

    :param shape: Shape of the picture segment resize
    :type shape: int

    :param input_path: path of the folder where the pictures are store
    :type input_path: str list
    :param cuts: #
    :return:
    """
    images = []
    for files_path in input_path:

        files = os.listdir(files_path)  # TODO paths
        for f in files:
            file_path = f'{files_path}/{f}'
            im = cv2.imread(file_path)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
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
def niegbors(number: int, number_sectors: int) -> [int, int, int, int]:
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
