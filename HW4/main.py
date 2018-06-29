###
"""
autors = Alejandro Moscoso
         Zac

"""
import keras
from models.Lenguage import create_labels_rnn, load_imbd
from models.ReviewGenerator import ReviewGenerator
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#%%

Data, Labels, word_to_id, id_to_word = load_imbd(5_000, 50)
sentiment = np.ones([Data.shape[0], 50]) * np.reshape(Labels, [Labels.size, 1])
L_rnn = create_labels_rnn(Data)
a = ReviewGenerator(v_size=5_000, review_len=50, l_s_t_m_state_size=512)
# %%
filepath = "data/{epoch:02d}-{loss:.4f}.h5"
tf_log_path = "/tf_log/rnn"

tbCallBack = keras.callbacks.TensorBoard(log_dir=tf_log_path,
                                         histogram_freq=0,  write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=1, min_lr=0.00001)
#%%

a.model.fit([Data, sentiment], L_rnn, 10, 100, validation_split=0.2, verbose=2, callbacks=[tbCallBack],
            initial_epoch=50)
# a.model.save('Data/Model_rnn2.h5')

