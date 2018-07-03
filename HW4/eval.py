#
"""
Authors :        Zachary Bamberger
                Alejandro Moscoso
summary :       This Scrip is use for evaluating
                and debugging our model, here we generate
                the sequences and score then using Blue

"""

# External Modules
import sys
from tqdm import tqdm
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
# Internal Modules
import models.Lenguage as lg
import models.ReviewGenerator as rg
from models.Lenguage import pd2list
import models.BLEU as bl

# MACROS
WORD_COUNT = 18_000
REVIEW_LENGHT = 80

# %%
# Get information necessary to load our model from memory.
try:
    model_path = sys.argv[1]
except IndexError as e:
    # TODO: configure this default path based on your system and where you stored the model
    model_path = "data/96-3.4122.h5"

Data, Labels, word_to_id, id_to_word = lg.load_imbd(1, 1)
trained_model = rg.ReviewGenerator(load_path=r"data/200-2.4376.h5")

# %%
# Define the column vector to represent positive and negative sentiments. These will be used when generating
# sentences from our model.
# TODO: make sure these are initialized correctly.
positive_sentiment = np.ones(shape=[1, REVIEW_LENGHT])
negative_sentiment = np.zeros(shape=[1, REVIEW_LENGHT])

mix = negative_sentiment.copy()
mix[0, 10:] = 1

# Generate 25 positive and 25 negative sentences
positive_sentences = []
negative_sentences = []
# %%
# Print one sequence

p = trained_model.generate_text(seed='i love this'.split(), max_len=REVIEW_LENGHT, temperature=0.3,
                                word_sentiment=negative_sentiment, verbose=0)

print(' '.join([id_to_word[id_] for id_ in p.reshape(-1)]))
# %%

for _ in tqdm(range(10)):
    positive_sentences.append(
        trained_model.generate_text(seed=["<START>"], max_len=REVIEW_LENGHT, temperature=0.3,
                                    word_sentiment=positive_sentiment)
    )
    negative_sentences.append(
        trained_model.generate_text(seed=["<START>"], max_len=REVIEW_LENGHT, temperature=0.3,
                                    word_sentiment=negative_sentiment)
    )
# %%
# mix sentence
mix_sentences = []
for _ in tqdm(range(10)):
    mix_sentences.append(
        trained_model.generate_text(seed=["<START>"], max_len=REVIEW_LENGHT, temperature=0.3, word_sentiment=mix)
    )

    # %%
# Print the sentence
for sentance in mix_sentences:
    print(' '.join([id_to_word[id] if id != 0 else "" for id in sentance.reshape(-1)]))

# %%
# split data set into positive and negative datasets for the BLUE score
Data, Labels, word_to_id, id_to_word = lg.load_imbd(18_000, 80)
Sample_D, _, Sample_L, _ = train_test_split(Data, Labels, train_size=0.1)

positive_corpus = pd2list(Sample_D[Sample_L == 1], id_to_word)
negative_corpus = pd2list(Sample_D[Sample_L == 0], id_to_word)

# %%

positive_sentences = pd2list(pd.DataFrame(positive_sentences), id_to_word)
negative_sentences = pd2list(pd.DataFrame(negative_sentences), id_to_word)

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
