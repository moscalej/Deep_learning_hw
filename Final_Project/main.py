import yaml
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from Final_Project.models.Algorithm_sketch import Puzzle
from Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train
from Final_Project.models.discriminator import create_arch_discrimitor, border_compare, create_call_backs, fire_disc
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Final_Project.models.Algorithm_sketch import get_prob_dict
import seaborn as sns
# %%
from Final_Project.models.evaluate import evaluate

# l =evaluate(file_dir=r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Deep Learning\HW\hw_repo\Deep_learning_hw\Final_Project\data\example\\')


# # %%
SHAPE = 64
path_train = [r'Final_Project\data\images', r'Final_Project\data\documents']
# path_train = [r'Final_Project\data\images']
# path_train = [r'Final_Project\data\toy_data']
path_validation = r'Final_Project\data\images_small'

path_D = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\documents'
processed_train = pre_process_data(path_train, shape=SHAPE, cuts=4)
# processed_validation = pre_process_data([path_validation], shape=SHAPE, cuts=5)
# trainX_0,trainX_1, trainY = processed2train_2_chanel(a,5)
trainX_0, trainX_1, trainY = processed2train_2_chanel(processed_train, 4)
# valX, valY = processed2train(processed_validation, 5, mode='validation')
# %%
model = create_arch_discrimitor(ud_lr='ds', weight_decay=8e-5, input_size=SHAPE)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.8,
                              patience=3,
                              min_lr=0.000001,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)


# %%
def generator(trainX_0, trainX_1, trainY, batch_size):
    size = trainY.shape[0]
    index = 0
    end = batch_size
    trainX_0, _, trainX_1, _, trainY, _ = train_test_split(trainX_0, trainX_1, trainY, shuffle=True)
    while True:
        yield [trainX_0[index:end], trainX_1[index:end]], trainY[index:end]
        index = end
        end += batch_size
        if index >= size - 1:
            index = 0
            end = batch_size


# %%
model.fit(x=[trainX_0, trainX_1], y=trainY, batch_size=1024, epochs=20, validation_split=0.2, shuffle=True,
          callbacks=[reduce_lr])

# %%
y_hat = model.predict([trainX_0, trainX_1])
cm = sklearn.metrics.confusion_matrix(np.argmax(trainY, axis=1), np.argmax(y_hat, axis=1))
# %%
df = pd.DataFrame(trainY)
t = pd.Series(np.argmax(df, axis=1))
t = pd.Series(np.argmax(df.values, axis=1))
t.hist()
plt.show()
