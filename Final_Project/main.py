import yaml
from Deep_learning_hw.Final_Project.models.Algorithm_sketch import Puzzle
from Deep_learning_hw.Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train
from Deep_learning_hw.Final_Project.models.discriminator import create_arch_discrimitor,border_compare
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Deep_learning_hw.Final_Project.models.Algorithm_sketch import get_prob_dict
import seaborn as sns
# %%
from Deep_learning_hw.Final_Project.models.evaluate import evaluate
l =evaluate(file_dir=r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Deep Learning\HW\hw_repo\Deep_learning_hw\Final_Project\data\example\\')


# # %%
# SHAPE = 64
# path_train = r'Deep_learning_hw\Final_Project\data\images'
# path_validation = r'Deep_learning_hw\Final_Project\data\images_small'
#
# # path_D = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\documents'
# processed_train = pre_process_data([path_train], shape=SHAPE, cuts=5)
# processed_validation = pre_process_data([path_validation], shape=SHAPE, cuts=5)
# # trainX_0,trainX_1, trainY = processed2train_2_chanel(a,5)
# trainX, trainY = processed2train(processed_train, 5)
# valX, valY = processed2train(processed_validation, 5, mode='validation')
# # %%
# model = border_compare(input_size=SHAPE)
# #
# # #%%
# # # x_mean = np.mean(trainX, axis=(0, 1, 2))
# # # x_std = np.std(trainX, axis=(0, 1, 2))
# # # x_center = (trainX - x_mean) / (x_std + 1e-9)
# # #%%
# # model.fit(trainX, trainY, validation_data=[valX, valY], batch_size=512, epochs=10, verbose=2)
# #
# # %%
# y_hat = model.predict([trainX_0, trainX_1])
# cm = sklearn.metrics.confusion_matrix(np.argmax(trainY, axis=1), np.argmax(y_hat, axis=1))
#
# df = pd.DataFrame(trainY)
# t = pd.Series(np.argmax(df, axis=1))
# t = pd.Series(np.argmax(df.values, axis=1))
# t.hist()
# plt.show()
