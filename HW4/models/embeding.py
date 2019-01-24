
from models.Lenguage import load_imbd, data_generator_embeding
import keras.layers as kl
import keras.models as km
import keras

def embeding_network():
    standardModel = km.Sequential()
    standardModel.add(kl.Embedding(input_dim=1,output_dim=128))
    standardModel.add(kl.Dense(input_dim=128, output_dim=128 ))
    standardModel.add(kl.Dense(10_000, activation='softmax'))
    standardModel.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    return standardModel
Data, Labels, word_to_id, id_to_word = load_imbd(10_000, 200)
model = embeding_network()
model.fit_generator(data_generator_embeding(Data,batch_size=16,voc_size=10_000),50_000//1024,10)