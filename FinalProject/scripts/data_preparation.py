from models.DSC_M import DSCM
import numpy as np
#%%
from keras.preprocessing.image import ImageDataGenerator
img = ImageDataGenerator( featurewise_center=False,
                          samplewise_center=False,
                          featurewise_std_normalization=False,
                          samplewise_std_normalization=False,
                          zca_whitening=False,
                          zca_epsilon=1e-06,
                          rotation_range=0.1,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          brightness_range=None,
                          shear_range=0.0,
                          zoom_range=0.1,
                          channel_shift_range=0.0,
                          fill_mode='nearest',
                          cval=0.0,
                          horizontal_flip=True,
                          vertical_flip=True,
                          rescale=None,
                          preprocessing_function=None,
                          data_format=None,
                          validation_split=0.0)
#%%
img.fit(ds.images.reshape([1879,480,480,1]))
iter_image = img.flow(ds.images.reshape([1879,480,480,1]))
new_training_set = np.concatenate([next(iter_image) for _ in range(1000)],axis=0)
