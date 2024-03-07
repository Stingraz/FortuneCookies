# Author: Yiwei

import numpy as np
import pandas as pd
import spacy


# wouldn't it make more sense to calculate the distance to another dataset?
def calculate_similarity_within_generated_fortunes(sentences):
    # Load English model
    nlp = spacy.load("en_core_web_md")

    similarity_scores = []

    # vectorize all sentences
    vectorized_sentences = [nlp(sentence) for sentence in sentences]

    # Calculate similarity between pairs of sentences
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_score = vectorized_sentences[i].similarity(vectorized_sentences[j])
            similarity_scores.append(similarity_score)

    # Calculate the average similarity score
    average_similarity = np.mean(similarity_scores)
    return average_similarity
