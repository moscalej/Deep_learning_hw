###
"""
Authors :       Zachary Bamberger
                Alejandro Moscoso
summary :       This Scrip is use for Building
                and training our model, here we load the
                the sequences from imdb preprocess then, create
                a model and call backs for monitoring and optimize the
                training.
"""
import keras
from Deep_learning_hw.models.Lenguage import load_imbd, data_generator
from Deep_learning_hw.models.ReviewGenerator import ReviewGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# %%
WORD_COUNT = 18_000
REVIEW_LENGHT = 80

Data, Labels, word_to_id, id_to_word = load_imbd(WORD_COUNT, REVIEW_LENGHT)

a = ReviewGenerator(v_size=WORD_COUNT, review_len=REVIEW_LENGHT, l_s_t_m_state_size=1024)
# %%
filepath = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW4\checkpoint\word_level\{epoch:02d}-{loss:.4f}.h5"
tf_log_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW4\TF\word_level"

tbCallBack = keras.callbacks.TensorBoard(log_dir=tf_log_path,
                                         histogram_freq=0, write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0, )

checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True, mode='min',save_weights_only=False)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8,
                              patience=30, min_lr=0.00001,
                              embeddings_layer_names=None, embeddings_metadata=None)
a.model.summary()
# %%

generator = data_generator(Data.values, Labels.values, 128, voc_size=WORD_COUNT)

a.model.fit_generator(generator, steps_per_epoch=25_000 // 128, epochs=1000,
                      verbose=1, callbacks=[tbCallBack, checkpoint, reduce_lr],
                      use_multiprocessing=False, shuffle=True, initial_epoch=0)

a.model.save(r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW4\checkpoint\word_level.h5')
