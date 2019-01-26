import yaml
from Final_Project.models.Algorithm_sketch import Puzzle
from Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train
from Final_Project.models.discriminator import create_arch_discrimitor,border_compare
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Final_Project.models.Algorithm_sketch import get_prob_dict
import seaborn as sns

# %%
SHAPE = 40
path = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\images'
# path_D = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\documents'
a = pre_process_data([path], shape=SHAPE, cuts=5)
# trainX_0,trainX_1, trainY = processed2train_2_chanel(a,5)
trainX, trainY = processed2train(a, 5)
# %%
model = border_compare(input_size=SHAPE)

#%%
x_mean = np.mean(trainX, axis=(0, 1, 2))
x_std = np.std(trainX, axis=(0, 1, 2))
x_center = (trainX - x_mean) / (x_std + 1e-9)
#%%
model.fit(x_center, trainY, batch_size=512, epochs=10, verbose=2, validation_split=0.2)

# %%
y_hat = model.predict([trainX_0, trainX_1])
cm = sklearn.metrics.confusion_matrix(np.argmax(trainY, axis=1), np.argmax(y_hat, axis=1))

df = pd.DataFrame(trainY)
t = pd.Series(np.argmax(df, axis=1))
t = pd.Series(np.argmax(df.values, axis=1))
t.hist()
plt.show()
