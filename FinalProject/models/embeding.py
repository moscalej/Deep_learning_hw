"""
#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
"""
import keras.layers as kl
import keras.models as km
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from FinalProject.models.Preprocesses import for_embeding, pre_proccess_data


def create_model(weight_decay = 6e-3):
    "the idea is to encode pictures to global position "
    in_photo = kl.Input((32,32,1))
    encoder = kl.Conv2D(32,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(in_photo)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Conv2D(32,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) # 16

    encoder = kl.Conv2D(64,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) # 8
    encoder = kl.Conv2D(96,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) # 4
    encoder = kl.Conv2D(128,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Conv2D(128,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder) #2
    encoder = kl.Flatten()(encoder)
    encoder = kl.Dense(128,kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.4)(encoder)
    encoder = kl.Dense(64,kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.4)(encoder)

    encoder = kl.Dense(3,activation='tanh')(encoder)
    model = km.Model(inputs=in_photo, outputs=encoder)
    model.compile(optimizer='adam',loss='mse')
    return model
#%%
paths = [r'C:/Users/amoscoso/Documents/Technion/deeplearning/Deep_learning_hw/FinalProject/data/images',
         r'C:/Users/amoscoso/Documents/Technion/deeplearning/Deep_learning_hw/FinalProject/data/documents']

t = pre_proccess_data(paths,
                      cuts=5,
                      shape=32)
x_train , y_train = for_embeding(t)
print(x_train.shape)
#%%
filepath = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\train\check_pont\embding_v1\embding_v4.{epoch:02d}-{val_loss:.2f}.hdf5'
tf_fp = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\train\tf\embding_v4'
tbCallBack = TensorBoard(log_dir=tf_fp,
                         histogram_freq=0,
                         batch_size=32,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)
modelcheck = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=0, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=10)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.8,
                              patience=3,
                              min_lr=0.000001,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
ear_s =EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#%%
img = ImageDataGenerator(featurewise_center=False,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         rotation_range=15,
                         width_shift_range=0.13,
                         height_shift_range=0.13,
                         brightness_range=None,
                         shear_range=0.0,
                         zoom_range=0.1,
                         channel_shift_range=0.0,
                         fill_mode='nearest',
                         cval=0.0,
                         horizontal_flip=False,
                         vertical_flip=False,
                         rescale=None,


                         )

img.fit(x_train)
#%%
model = create_model()
model.summary()
#%%
x_train_,x_test, y_train_,y_test = train_test_split(x_train,y_train,test_size=0.1)
history_fully = model.fit_generator(img.flow(x_train_, y_train_, batch_size=1024), steps_per_epoch=48,
                                    shuffle=True,
                                    epochs=500,
                                    initial_epoch=120,
                                    validation_data=[x_test,y_test],
                                     callbacks=[tbCallBack, reduce_lr,modelcheck])
#%%

results = model.fit(x_train,
                    y_train
                    ,1024,
                    1000,
                    validation_split=0.2,
                    callbacks=[tbCallBack,modelcheck,reduce_lr,ear_s])
