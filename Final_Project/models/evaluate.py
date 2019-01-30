# %%
import os
import cv2
from Deep_learning_hw.Final_Project.models.Algorithm_sketch import predict_2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# %%

def predict(images, showimage=True):  # todo showimage=False


    labels = predict_2(images)
    # if showimage:
    #     crop_num = len(labels)
    #     axis_size = int(np.sqrt(crop_num))
    #     one2two_dims_ = np.reshape(np.array([i for i in range(crop_num)]), (axis_size, axis_size))
    #     ind_one2two = {i: tuple(list(map(int, list(np.where(one2two_dims_ == i))))) for i in range(crop_num)}
    #     fig = plt.figure(figsize=(axis_size, axis_size))
    #     ax = [plt.subplot(axis_size, axis_size, i + 1) for i in range(crop_num)]
    #     for ind, label in enumerate(labels):
    #         current_image = images[label]
    #         ax[label].imshow(current_image)
    #         plt.imshow(current_image)
    #         ax[label].set_aspect('equal')
    #         ax[label].set_xticklabels([])
    #         ax[label].set_yticklabels([])
    #         # currentX, currentY = ind_one2two[label]
    #         # plt.subplot(axis_size, axis_size, 1+ind)
    #         # plt.subplot(len(labels), currentX+1, currentY+1)
    #     fig.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()
    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    # print("True order:")
    orig_order = [file.split('_')[2].split('.')[0] for file in files]
    # files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    print("Ground_truth->predicted:")
    print([f"{orig}-> {Y[ind]}" for ind, orig in enumerate(orig_order)])


    return Y
