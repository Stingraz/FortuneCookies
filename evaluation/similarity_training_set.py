# Author: Yiwei

import numpy as np
import pandas as pd
import spacy


def calculate_similarity_to_training_set(training_fortunes, sentences):
    similarity_scores = []
    sentences_scores = {}

    nlp = spacy.load("en_core_web_md")

    training_vectors = [nlp(sentence).vector for sentence in training_fortunes]
    average_training_vector = sum(training_vectors) / len(training_vectors)

    for sentence in sentences:
        sentence_vector = nlp(sentence).vector
        similarity_score = np.dot(sentence_vector, average_training_vector) / (
                np.linalg.norm(sentence_vector) * np.linalg.norm(average_training_vector))
        similarity_scores.append(similarity_score)
        sentences_scores[sentence] = similarity_score

    dataframe = pd.DataFrame(list(sentences_scores.items()),
                             columns=['Sentence', 'Similarity to average training sentence'])
    average_similarity = np.mean(similarity_scores)

    return average_similarity, dataframe[dataframe.columns[-1]]
