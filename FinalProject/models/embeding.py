"""
#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
"""
import keras.layers as kl
import keras.models as km
import keras
from FinalProject.models.Preprocesses import for_embeding, pre_proccess_data


def create_model(weight_decay = 1e-5):
    "the idea is to encode pictures to global position "
    in_photo = kl.Input((32,32,1))
    encoder = kl.Conv2D(16,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(in_photo)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) # 16
    encoder = kl.Conv2D(32,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) # 8
    encoder = kl.Conv2D(64,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 4
    encoder = kl.Conv2D(128,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) #2
    encoder = kl.Flatten()(encoder)
    encoder = kl.Dense(128,kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.Activation('tanh')(encoder)
    encoder = kl.Dense(3,activation='softmax')(encoder)
    model = km.Model(inputs=in_photo, outputs=encoder)
    model.compile(optimizer='adam',loss='mse')
    return model
model = create_model()
model.summary()
paths = [r'C:/Users/amoscoso/Documents/Technion/deeplearning/Deep_learning_hw/FinalProject/data/images',
         r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\documents']
t = pre_proccess_data(paths,
                      cuts=5,
                      shape=32)
x_train , y_train = for_embeding(t)
results = model.fit(x_train,y_train,1024,10,validation_split=0.2)
