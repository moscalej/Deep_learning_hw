from keras.datasets import imdb
import pandas as pd
from keras.preprocessing import sequence


# load the dataset but only keep the top words, zero the rest. Introduce special tokens.
from keras.utils import to_categorical
import numpy as np


def load_imbd(top_words=5000, max_length=150):
    (Data, Labels), _ = imdb.load_data(num_words=top_words)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<OOV>"] = 2
    id_to_word = {v: k for k, v in word_to_id.items()}
    Data = sequence.pad_sequences(pd.Series(Data).values, maxlen=max_length, padding='post', truncating='post')
    return pd.DataFrame(Data), pd.Series(Labels), word_to_id, id_to_word


def tranaslte(data_id_rows, id_to_word):
    foo = lambda x: id_to_word[x]
    return data_id_rows.apply(foo)

def create_labels_rnn(Y):
    a = to_categorical(np.roll(Y, -1))
    return a

if __name__ == '__main__':
    Data, Labels, word_to_id, id_to_word = load_imbd(5000, 100)
    sentiment = np.ones([Data.shape[0],100])* np.reshape(Labels,[Labels.size,1])
    L_rnn = create_labels_rnn(Data)

