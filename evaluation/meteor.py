# Author: Michelle Fribance

import numpy as np
import pandas as pd
import nltk
from nltk.translate import meteor_score


def calculate_single_meteor(reference_data, fortune_to_evaluate):
    # Calculate METEOR score for a single generated fortune:
    fortune_meteor_score = meteor_score.meteor_score(reference_data, fortune_to_evaluate)
    return fortune_meteor_score


def calculate_dataset_meteor(training_fortunes_list, sentences):
    try:
        # Check if punkt is found
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # If not found, download punkt
        nltk.download('punkt')

    # nltk.download('wordnet')

    # Strip whitespaces from reference data
    training_fortunes_list = [fortune.strip() for fortune in training_fortunes_list]
    # Tokenize the reference data
    training_fortunes_list_tokenized = [nltk.word_tokenize(fortune.lower()) for fortune in training_fortunes_list]

    # Tokenize the generated fortunes
    fortunes_to_evaluate_tokenized = [nltk.word_tokenize(fortune.lower()) for fortune in sentences]

    # Calculate METEOR score for each fortune in fortunes_to_evaluate
    meteor_scores = [
        calculate_single_meteor(reference_data=training_fortunes_list_tokenized, fortune_to_evaluate=fortune) for
        fortune in fortunes_to_evaluate_tokenized]

    meteor_scores = pd.DataFrame(meteor_scores)

    return np.mean(meteor_scores), meteor_scores
