import keras
from keras.models import Model, load_model
from keras.layers import *


class ReviewGenerator:
    def __init__(self, load_path=None):
        if load_path is not None:
            self.model = load_model(load_path)
        else:
            self.model = self._build_model()

    def _build_model(self, v_size=5000, review_len=100):
        VOCABULARY_SIZE = v_size
        LSTM_state_size = review_len

        # sequence model Input and Embeding
        inputs1 = Input(shape=(review_len,))
        in_and_embedding = Embedding(VOCABULARY_SIZE, LSTM_state_size, mask_zero=True)(inputs1)
        in_and_embedding = Dropout(0.3)(in_and_embedding)

        # Sensitivity Input
        input2 = Input((1,))
        sentiment_flow = Dense(review_len, activation='relu')(input2)

        merge = add([sentiment_flow, in_and_embedding])

        merged_flow = LSTM(LSTM_state_size, return_sequences=True)(merge)
        merged_flow = LSTM(LSTM_state_size, return_sequences=True)(merged_flow)

        merged_flow = Dropout(0.3)(merged_flow)
        merged_flow = TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax'))(merged_flow)
        model = Model([inputs1, input2], merged_flow)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def _build_callbacks(self):
        pass

    def fit(self):
        return self.model.fit()


if __name__ == '__main__':
    a = ReviewGenerator()
