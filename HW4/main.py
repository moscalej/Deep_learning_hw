###
"""
autors = Alejandro Moscoso
         Zac

"""
import keras

from models.Lenguage import create_labels_rnn, load_imbd
from models.ReviewGenerator import ReviewGenerator
import numpy as np

#%%

Data, Labels, word_to_id, id_to_word = load_imbd(5000, 100)
sentiment = np.ones([Data.shape[0],100])* np.reshape(Labels,[Labels.size,1])
L_rnn = create_labels_rnn(Data)
a = ReviewGenerator(v_size=5_000, review_len=100,l_s_t_m_state_size=64)

tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/gilshoshan/Documents/deep/Deep_learning_hw/HW4/Tf_log/rnn',
                                         histogram_freq=0,  write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None)


a.model.fit([Data, sentiment], L_rnn, 128, 100, validation_split=0.2,verbose=2,callbacks=[tbCallBack])
a.model.save('Data/Model_rnn.h5')

