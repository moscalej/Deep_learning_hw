

from keras.models import Model, load_model
from keras.layers import *
import numpy as np


class ReviewGenerator:
    def __init__(self, word2ind=None, ind2word=None, load_path=None, v_size=5000, review_len=100, l_s_t_m_state_size=8):
        if load_path is not None:
            self.model = load_model(load_path)
        else:
            self.model = self._build_model(v_size=v_size,
                                           review_len=review_len,
                                           l_s_t_m_state_size=l_s_t_m_state_size)
        self.word2ind=word2ind
        self.ind2word=ind2word
        # self.fit = self.model.fit

    def _build_model(self, v_size=5000, review_len=100, l_s_t_m_state_size=8):
        VOCABULARY_SIZE = v_size


        # sequence model Input and Embeding
        inputs1 = Input([review_len])
        in_and_embedding = Embedding(VOCABULARY_SIZE,
                                     l_s_t_m_state_size, mask_zero=True,
                                     input_length=review_len)(inputs1)
        in_and_embedding = Dropout(0.3)(in_and_embedding)

        #Sensitivity Input
        input2 = Input([review_len])
        # sentiment_flow = Dense(review_len, activation='relu')(input2)
        sentiment_flow = Reshape([review_len, 1])(input2)
        # Merge the inputs
        merge_layer = multiply([sentiment_flow, in_and_embedding])

        merged_flow = LSTM(l_s_t_m_state_size, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(
            merge_layer)

        merged_flow = LSTM(l_s_t_m_state_size, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)(
            merged_flow)

        out_final = TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax'))(merged_flow)
        model = Model(inputs=[inputs1, input2],outputs= out_final)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])

        return model

    def _build_callbacks(self):
        pass

    def sample(self, preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(self, seed=["the", 'movie', 'was'], max_len=300, temperature=0.5, word_sentiment=[], verbose=0):
        """Generate characters from a given seed"""
        result = np.zeros((1, max_len))
        result[0, 0] = self.word2ind["<START>"]
        next_res_ind = 1
        for s in seed:
            result[0, next_res_ind ] = self.word2ind[s]
            next_res_ind = next_res_ind + 1

        if verbose > 0: print(f'[ {" ".join(seed)}] ')

        nex_word = seed[-1]
        while nex_word != "<PAD>" and next_res_ind < max_len:
            self.model.reset_states()
            y = self.model.predict_on_batch([result, word_sentiment])[0][next_res_ind - 1]
            y[2] = 0

            nex_word = self.sample(y, temperature)

            result[0, next_res_ind] = nex_word
            next_res_ind = next_res_ind + 1
            if verbose > 0: print(self.ind2word[nex_word])
        return result.reshape(-1)


if __name__ == '__main__':
    pass
