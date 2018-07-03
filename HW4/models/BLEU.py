"""
Authors :     Zachary Bamberger
             Alejandro Moscoso
summary :    This Module is in charge of :
             defining the class
                        BLEU
                            witch will be use for scoring
"""

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


class BLEU:
    def __init__(self, reference_sentences, candidate_sentences=[], ngram_weights=(1, 0, 0, 0)):
        """

        :param reference_sentences: inputted list of sentences. Our corpus
        :param candidate_sentences: sentences whos scores we are trying to get against the corpus.
        :param ngram_weights: the weight of the n gram model for this bleu score. default is a score
        based entirely on the unigram model
        """
        self.reference_sentences = reference_sentences
        self.candidate_sentences = candidate_sentences
        self.ngram_weights = ngram_weights

    def get_sentence_score(self, candidate_sentence):
        """

        :param candidate_sentence: The sentence for which we are generating a bleu score
        :return: the bleu score for this sentence given the corpus (stored in self) and the ngram weights tuple.
        """
        return sentence_bleu(self.reference_sentences, candidate_sentence, weights=self.ngram_weights)

    def get_cumulative_candidate_scores(self):
        """

        :return: the bleu score for list of sentence results (stored in self) given the corpus
         (also stored in self) and the ngram weights tuple.
        """
        return corpus_bleu([self.reference_sentences], [self.candidate_sentences], weights=self.ngram_weights)

    def get_mean_bleu_score(self):
        """

        :return: The mean bleu score for all candidate sentences measured against the provided corpus.
        """
        acc = 0
        for candidate in self.candidate_sentences:
            score = sentence_bleu(self.reference_sentences, candidate)
            acc += score
        return acc / len(self.candidate_sentences)


def calculate_bleu(candidate_sentences, reference_sentences):
    """
    return the mean bleu score for the batch and a dictionary of the bleu scores for each sentence.
    :param candidate_sentences:
    :param reference_sentences:
    :return: the mean accuracy and a list of the bleu scores of every sentence (in the same order as the sentences).
    """

    acc = 0
    scores = []
    for candidate in candidate_sentences:
        score = sentence_bleu(reference_sentences, candidate)
        acc += score
        scores.append(score)
    return (acc / len(candidate_sentences)), scores
