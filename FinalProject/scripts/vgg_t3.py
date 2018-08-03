"""
Authors :       Zachary Bamberger
                Alejandro Moscoso
"""
import os
from models.Data_set_creator import DSC
from models.dnn import benchmark_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

if "Zach" in os.environ.get('USERNAME'):
    image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\images"
    #  TODO define this paths
    tf_log_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\TB\vgg_bm'
    check_point_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\check_point\vgg19_bm_t3'

else:
    image_path = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\images"
    tf_log_path = r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\TB\vgg_bm'
    check_point_path = \
        r'D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\check_point\vgg19_bm_t3\{epoch:02d}-{loss:.4f}.h5'



#%%
#  Data generators
t_2_dataset = DSC(images_path=image_path, t_value=2)
t_4_dataset = DSC(images_path=image_path, t_value=4)
t_5_dataset = DSC(images_path=image_path, t_value=5)

#%%
tbCallBack = TensorBoard(log_dir=tf_log_path,
                         histogram_freq=0, write_graph=True,
                         write_grads=False, write_images=False, embeddings_freq=0, )

checkpoint = ModelCheckpoint(check_point_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=2, min_lr=0.00001,
                              embeddings_layer_names=None, embeddings_metadata=None)


# %%

model =  benchmark_model(number_lstm=25, state_size=512, dense_size=512, vgg_trainable=False, optimizer='adam')
model.summary()
# %%

generator = t_5_dataset.generate_batch(128)

model.fit_generator(generator, steps_per_epoch=1300 // 128, epochs=10,
                      verbose=1, callbacks=[tbCallBack, checkpoint, reduce_lr],
                      use_multiprocessing=False, shuffle=True, initial_epoch=0)













