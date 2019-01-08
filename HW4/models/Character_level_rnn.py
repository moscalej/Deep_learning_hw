import numpy as np
from keras.layers import CuDNNLSTM, Dropout

from keras.layers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
import pandas as pd
import string

df = pd.read_csv('data.csv')
tweets = [t for t in df.Text.tolist() if type(t)==str]
for i in range(10):
    print(tweets[i], '\n')

# generic vocabulary
characters = ['<PAD>', '<START>', '<EOS>'] + list(string.printable)  # we add pad, start, and end-of-sentence
characters.remove('\x0b')
characters.remove('\x0c')

VOCABULARY_SIZE = len(characters)
char2ind = {c:i for i,c in enumerate(characters)}
print("vocabulary len = %d" % VOCABULARY_SIZE)
print(characters)
tweets_tokenized = [[char2ind['<START>']] + [char2ind[c] for c in tweet if c in char2ind] + [char2ind['<EOS>']] for tweet in tweets]
x_train = np.array(sequence.pad_sequences(tweets_tokenized))
y_train = np.roll(x_train, -1, axis=-1)  # we want to predict the next character
y_train[:, -1] = char2ind['<EOS>']

x_train = np.array([to_categorical(x, num_classes=VOCABULARY_SIZE) for x in x_train])
y_train = np.array([to_categorical(y, num_classes=VOCABULARY_SIZE) for y in y_train])
print(x_train.shape)
print(y_train.shape)

LSTM_state_size = 512

model = Sequential()
model.add((CuDNNLSTM(LSTM_state_size, return_sequences=True, input_shape=x_train.shape[1:])))
model.add((CuDNNLSTM(LSTM_state_size, return_sequences=True)))
model.add(Dropout(0.3))
model.add((CuDNNLSTM(LSTM_state_size, return_sequences=True)))
model.add((CuDNNLSTM(LSTM_state_size, return_sequences=True)))
model.add(Dropout(0.3))
# model.add((CuDNNLSTM(LSTM_state_size, return_sequences=True)))
model.add(TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train,
          validation_split=0.1,
          epochs=40, batch_size=128)


def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed="I am", max_len=300, diversity=0.5):
    """Generate characters from a given seed"""
    result = np.zeros((1,) + x_train.shape[1:])
    result[0, 0, char2ind['<START>']] = 1
    next_res_ind = 1
    for s in seed:
        result[0, next_res_ind, char2ind[s]] = 1
        next_res_ind = next_res_ind + 1

    print("[" + seed + "]", end='')

    next_char = seed[-1]
    while next_char != '<EOS>' and next_res_ind < max_len:
        model.reset_states()
        y = model.predict_on_batch(result)[0][next_res_ind - 1]
        next_char_ind = sample(y, temperature=diversity)
        next_char = characters[next_char_ind]
        result[0, next_res_ind, next_char_ind] = 1
        next_res_ind = next_res_ind + 1
        print(next_char, end='')
    print()


for i in range(5):
    generate_text(
        model,
        seed="I am here to "
    )
    print()