

#  External modules
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numba import njit

def load_imbd(top_words=5000, max_length=150):
    """
    This function will load the data from Keras pad the the
    sequence and create Data frames from the labels and The data
    :param top_words:
    :param max_length:
    :return:
    """
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

# @njit()
def create_labels_rnn(Y, num_classes):
    a = to_categorical(np.roll(Y, -1), num_classes=num_classes)
    return a


def pd2list(Data, id_to_word):
    list = []
    for row in Data.iterrows():
        list.append([id_to_word[id] for id in row[1]])
    return list

# @njit()
def data_generator(Data, Labels, batch_size=128, voc_size=20_000):
    while 1:
        x_batch, _, y_batch, _ = train_test_split(Data, Labels, train_size=batch_size,shuffle=True)
        sentiment = np.ones(x_batch.shape) * np.reshape((2*y_batch-1), [y_batch.size, 1])
        L_rnn = create_labels_rnn(x_batch, voc_size)
        yield [x_batch, sentiment], L_rnn


def data_generator_embeding(Data, batch_size=128, voc_size=20_000):
    tensor = Data.values
    values = []
    for i in [2, 0, 1, 3, 4]:
        values.append(np.roll(tensor, i, axis=1).reshape(-1))

    data = values[0]
    val = values[1:]
    labels = np.array(values[1:])
    while 1:
        rand_labels = np.apply_along_axis(np.random.choice, 0, labels)
        for mini_bath in range(batch_size, data.shape[0] // batch_size, batch_size):
            x_batch = data[mini_bath - batch_size:mini_bath]
            y_batch = rand_labels[mini_bath - batch_size:mini_bath]
            # x_batch_one = to_categorical(x_batch, num_classes=voc_size).reshape([batch_size,voc_size,1])
            y_batch_one = to_categorical(y_batch, num_classes=voc_size).reshape([batch_size,voc_size,1])
            yield (x_batch, y_batch_one)


if __name__ == '__main__':
    pass
    # Data, Labels, word_to_id, id_to_word = load_imbd(20_000, 200)
