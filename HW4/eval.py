import os
import sys
from tqdm import tqdm
import models.Lenguage as lg
import models.ReviewGenerator as rg
import models.BLEU as bl
import numpy as np
from keras.models import load_model

WORD_COUNT = 18_000
REVIEW_LENGHT = 80

# %%
# Get information necessary to load our model from memory.
try:
    model_path = sys.argv[1]
except IndexError as e:
    # TODO: configure this default path based on your system and where you stored the model
    model_path = "data/96-3.4122.h5"

# Use same initialization as used in training the model. Load trained model from memory.
Data, Labels, word_to_id, id_to_word = lg.load_imbd(1, 1)
trained_model = rg.ReviewGenerator(v_size=WORD_COUNT, review_len=REVIEW_LENGHT,
                                   l_s_t_m_state_size=512,
                                   ind2word=id_to_word, word2ind=word_to_id)
trained_model.model = load_model(r"data/200-2.4376.h5")

# %%
# Define the column vector to represent positive and negative sentiments. These will be used when generating
# sentences from our model.
# TODO: make sure these are initialized correctly.
positive_sentiment = np.ones(shape=[1, REVIEW_LENGHT])
negative_sentiment = np.zeros(shape=[1, REVIEW_LENGHT])

mix = positive_sentiment.copy()


# Generate 25 positive and 25 negative sentences
positive_sentences = []
negative_sentences = []
# %%

p = trained_model.generate_text(seed='this is'.split(),
                                word_sentiment=negative_sentiment, max_len=REVIEW_LENGHT,
                                diversity=0.3,
                                verbose=1)


# %%

for _ in tqdm(range(25)):
    positive_sentences.append(
        trained_model.generate_text(word_sentiment=positive_sentiment, max_len=200, diversity=0.7)
    )
    negative_sentences.append(
        trained_model.generate_text(word_sentiment=negative_sentiment, max_len=200, diversity=0.7)
    )

# %%
# split data set into positive and negative datasets
Data, Labels, word_to_id, id_to_word = lg.load_imbd(5000, 100)
positive_corpus = []
negative_corpus = []
for ind in range(len(Labels)):
    # TODO: compare the label to the way in which it is marked as positive. Whether that be a string or an integer.
    # You will probably need to modify the sentence below
    if Labels[ind] == "POSITIVE":
        positive_corpus.append(Data[ind])
    else:
        negative_corpus.append(Data[ind])

# %%
# Start with unigram model. This will yield best results.
# Once we know things are working properly, check bigram and trigram models as well.
positive_bleu = bl.BLEU(reference_sentences=positive_corpus, candidate_sentences=positive_sentences)
negative_bleu = bl.BLEU(reference_sentences=negative_corpus, candidate_sentences=negative_sentences)

# %%
# Print out positive bleu scores in various formats.
# print("Positive bleu cumulative score: " + str(positive_bleu.get_cumulative_candidate_scores()))
print("Positive bleu mean score: " + str(positive_bleu.get_mean_bleu_score()))
print("Positive sentence bleu scores:")
for ps in positive_sentences:
    print(ps, positive_bleu.get_sentence_score(ps))

# Print out negative bleu scores in various formats.
# print("Negative bleu cumulative score: " + str(negative_bleu.get_cumulative_candidate_scores()))
print("Negative bleu mean score: " + str(negative_bleu.get_mean_bleu_score()))
print("Negative sentence bleu scores:")
for ns in positive_sentences:
    print(ns, negative_bleu.get_sentence_score(ns))
