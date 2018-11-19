"""
Authors :       Zachary Bamberger
                Alejandro Moscoso
"""
import numpy as np
import os
from models.DSGK import DSGk
from models.rnn_clasic import benchmark_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Determine the right path to the images.
if "Zach" in os.environ.get('USERNAME'):
    image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\images"
    #  TODO define this paths
    tf_log_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\TB\vgg_bm'
    check_point_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\check_point\vgg16_t5_f'

else:
    image_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\images"
    tf_log_path = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\TB\vgg16_t4_f'
    check_point_path = \
        r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\check_point\vgg16_t4_f\{epoch:02d}-{loss:.4f}.h5'



#%%

t_4_gen = DSGk(images_path=image_path, t_value=5)

#%%
tbCallBack = TensorBoard(log_dir=tf_log_path,
                         histogram_freq=0, write_graph=True,
                         write_grads=False, write_images=False, embeddings_freq=0, )

checkpoint = ModelCheckpoint(check_point_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=3, min_lr=0.000001,
                              embeddings_layer_names=None, embeddings_metadata=None)


# %%

model = benchmark_model(number_lstm=16, state_size=1024, dense_size=1024, vgg_trainable=True, optimizer='adam')
model.summary()

# %%
generator = t_4_gen.generate_batch(batch_size=4)

#%%
model.fit_generator(generator, steps_per_epoch=1879//4 , epochs=150,
                      verbose=1, callbacks=[tbCallBack, checkpoint, reduce_lr],
                      use_multiprocessing=False, shuffle=True, initial_epoch=0)



