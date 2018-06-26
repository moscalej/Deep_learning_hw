from nltk.translate.bleu_score import sentence_bleu


def calculate_bleu(candidate_sentences, reference_sentences):
    """
    return the mean bleu score for the batch and a dictionary of the bleu scores for each sentence.
    :param candidate_sentences:
    :param reference_sentences:
    :return: the mean accuracy and a dictionary mapping hashed sentences to their bleu scores.
    """

    acc = 0
    scores = {}
    for candidate in candidate_sentences:
        score = sentence_bleu(reference_sentences, candidate)
        acc += score
        scores[candidate] = score
    return (acc / len(candidate_sentences)), scores
