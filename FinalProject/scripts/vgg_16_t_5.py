"""
Authors :       Zachary Bamberger
                Alejandro Moscoso
"""
import numpy as np
import os
from models.DSC_M import DSCM
from models.rnn_clasic import benchmark_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Determine the right path to the images.
if "Zach" in os.environ.get('USERNAME'):
    image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\images"
    #  TODO define this paths
    tf_log_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\TB\vgg_bm'
    check_point_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\check_point\vgg19_bm_t3'

else:
    image_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\images"
    tf_log_path = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\TB\vgg16_bm_t5'
    check_point_path = \
        r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\check_point\vgg16_bm_t5\{epoch:02d}-{loss:.4f}.h5'



#%%
#  Data generators
# t_2_dataset = DSC(images_path=image_path, t_value=2)
# t_4_dataset = DSCM(images_path=image_path, t_value=4)
t_5_dataset = DSCM(images_path=image_path, t_value=5)

#%%
tbCallBack = TensorBoard(log_dir=tf_log_path,
                         histogram_freq=0, write_graph=True,
                         write_grads=False, write_images=False, embeddings_freq=0, )

checkpoint = ModelCheckpoint(check_point_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=2, min_lr=0.000001,
                              embeddings_layer_names=None, embeddings_metadata=None)


# %%

model = benchmark_model(number_lstm=25, state_size=1024, dense_size=1024, vgg_trainable=True, optimizer='adam')
model.summary()
# %%
t_5_dataset_val = DSCM(images_path=r'data/test', t_value=5)
generator_val = t_5_dataset_val.generate_batch(64)
val_set = [next(generator_val) for _ in range(64)]
val_x = np.concatenate([x[0] for x in val_set],axis=0)
val_y = np.concatenate([x[1] for x in val_set],axis=0)
# %%

generator = t_5_dataset.generate_batch(16)
#%%
model.fit_generator(generator, steps_per_epoch=1879//4 , epochs=100,validation_data=[val_x,val_y],
                      verbose=1, callbacks=[tbCallBack, checkpoint, reduce_lr],
                      use_multiprocessing=False, shuffle=True, initial_epoch=3)



