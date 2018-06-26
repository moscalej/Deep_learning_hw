from nltk.translate.bleu_score import sentence_bleu


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
