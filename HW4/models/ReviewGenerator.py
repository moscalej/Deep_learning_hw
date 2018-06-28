import keras
from keras.models import Model, load_model
from keras.layers import *


class ReviewGenerator:
    def __init__(self, word2ind=None,ind2word=None,load_path=None, v_size=5000, review_len=100,l_s_t_m_state_size=8):
        if load_path is not None:
            self.model = load_model(load_path)
        else:
            self.model = self._build_model(v_size=v_size,
                                           review_len=review_len,
                                           l_s_t_m_state_size=l_s_t_m_state_size)
        self.word2ind=word2ind
        self.ind2word=ind2word

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
        sentiment_flow = Dense(review_len, activation='relu')(input2)
        sentiment_flow = Reshape([review_len, 1])(input2)
        # Merge the inputs
        merge_layer = concatenate([sentiment_flow, in_and_embedding], axis=2)

        merged_flow = LSTM(l_s_t_m_state_size + 64, return_sequences=True)(merge_layer)
        merged_flow = LSTM(l_s_t_m_state_size + 64, return_sequences=True)(merged_flow)
        merged_flow = LSTM(l_s_t_m_state_size + 64, return_sequences=True)(merged_flow)

        merged_flow = Dropout(0.3)(merged_flow)
        out_final = TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax'))(merged_flow)
        model = Model(inputs=[inputs1, input2],outputs= out_final)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])

        return model

    def _build_callbacks(self):
        pass

    def fit(self):
        return self.model.fit()

    def generate_text(self,model, seed=["I", "am"], max_len=300, diversity=0.5,word_sentiment=[]):
        """Generate characters from a given seed"""
        result = np.zeros((1,) + max_len)
        result[0, 0, self.word2ind['<START>']] = 1
        next_res_ind = 1
        for s in seed:
            result[0, next_res_ind, self.word2ind[s]] = 1
            next_res_ind = next_res_ind + 1

        print("[" + seed + "]", end='')

        # next_char = seed[-1]
        # while next_char != '<EOS>' and next_res_ind < max_len:
        #     model.reset_states()
        #     y = model.predict_on_batch(result)[0][next_res_ind - 1]
        #     next_char_ind = sample(y, temperature=diversity)
        #     next_char = characters[next_char_ind]
        #     result[0, next_res_ind, next_char_ind] = 1
        #     next_res_ind = next_res_ind + 1
        #     print(next_char, end='')
        # print()


if __name__ == '__main__':
    a = ReviewGenerator(v_size=5_000, review_len=100,l_s_t_m_state_size=64)
