import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.layers import LSTM, Dropout

from keras.layers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import *
from keras.preprocessing import sequence
import string

top_words = 30_000
max_length = 100
(X_train, sentiment), _ = imdb.load_data(num_words=top_words, maxlen=max_length)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
id_to_word = {v: k for k, v in word_to_id.items()}
X_train_words = []
for i in range(X_train.shape[0]):
    X_train_words.append(' '.join(id_to_word.get(w) for w in X_train[i]))
# %%
# generic vocabulary
characters = ['<PAD>', '<START>', '<EOS>'] + list(string.printable)  # we add pad, start, and end-of-sentence
characters.remove('\x0b')
characters.remove('\x0c')
# %%
VOCABULARY_SIZE = len(characters)
char2ind = {c: i for i, c in enumerate(characters)}
print("vocabulary len = %d" % VOCABULARY_SIZE)


# %%

def to_char_level(X_train_words, char2ind):
    reviews_tokenized = []
    for review in X_train_words:
        sentence = [char2ind['<START>']]
        sentence.extend([char2ind[c] for c in review if c in char2ind])
        sentence.append(char2ind['<EOS>'])
        reviews_tokenized.append(sentence)
    return reviews_tokenized


reviews_tokenized = to_char_level(X_train_words, char2ind)
# %%
x_train = np.array(sequence.pad_sequences(reviews_tokenized))
y_train = np.roll(x_train, -1, axis=-1)  # we want to predict the next character
y_train[:, -1] = char2ind['<EOS>']


# %%
# @njit()
def categorical_words(x_train):
    a = []
    for x in x_train:
        a.append(to_categorical(x, num_classes=VOCABULARY_SIZE))
    return np.array(a)


x_train_cat = categorical_words(x_train)
y_train_cat = categorical_words(y_train)
print(x_train_cat.shape)
print(y_train_cat.shape)
# %%
LSTM_state_size = 512


def creat_carracter_level(LSTM_state_size, shape, optimizer='rmsprop', voc_size=101):
    in_1 = Input(shape)
    in_2 = Input(shape[:-1] +(1,))
    den = multiply([in_1, in_2])
    flow = CuDNNLSTM(LSTM_state_size, return_sequences=True)(den)
    flow = Dropout(0.3)(flow)
    flow = CuDNNLSTM(LSTM_state_size, return_sequences=True)(flow)
    flow = Dropout(0.3)(flow)
    flow = CuDNNLSTM(LSTM_state_size, return_sequences=True)(flow)
    flow = Dropout(0.3)(flow)
    flow = CuDNNLSTM(LSTM_state_size, return_sequences=True)(flow)
    flow = Dropout(0.3)(flow)
    flow = TimeDistributed(Dense(voc_size, activation='softmax'))(flow)
    model = Model([in_1, in_2], flow)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


model = creat_carracter_level(512, x_train_cat.shape[1:], optimizer='rmsprop', voc_size=VOCABULARY_SIZE)
# %%
# Call Backs
Tf_log = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW4\TF\CarL_v01'
tbCallBack = TensorBoard(log_dir=Tf_log,
                         histogram_freq=0,
                         batch_size=32,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=2,
                              min_lr=0.000001,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
# %%
sent = np.array([sentiment for _ in range(x_train_cat.shape[1])]).T.reshape((5736, 634,1))
model.fit([x_train_cat, sent],
          y_train_cat,
          validation_split=0.2,
          epochs=200,
          batch_size=128,
          callbacks=[tbCallBack, reduce_lr])
model.save(r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW4\models\cl.h5')


# %%
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
    global characters
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
