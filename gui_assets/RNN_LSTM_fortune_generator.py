# Author: Sarah

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Activation, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import os

# List of seed sentences
seed_sentences = [
    "you should beware of",
    "happiness comes from",
    "the happiness brings",
    "meaning of life is",
    "the journey of life",
    "success is achieved",
    "knowledge is the key",
    "friendship is like",
    "love is the greatest",
    "the world needs more",
    "life is a journey",
]

def generate_sentence(model_path, char_index_path, temperature=1.0, SEQLEN=20):
    """
    Generate a sentence based on a seed sentence using the provided model and dictionaries.

    Parameters:
    - model_path: Path to the .pkl file containing the trained RNN model.
    - char_index_path: Path to the .pkl file containing char2index and index2char dictionaries.
    - temperature: Controls the randomness of the predictions (higher values produce more random results).
    - SEQLEN: Length of the sequences used in training the model.

    Returns:
    - generated_sentence: The generated sentence.
    """

    # Load the model
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load char2index and index2char dictionaries
    try:
        with open(char_index_path, 'rb') as char_index_file:
            char2index, index2char = pickle.load(char_index_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Char index file not found at {char_index_path}")

    nb_chars = len(char2index)

    # Choose a random seed sentence
    seed_sentence = np.random.choice(seed_sentences)

    test_chars = seed_sentence
    generated_sequence = test_chars

    for i in range(500):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        preds = model.predict(Xtest, verbose=0)[0]
        next_index = sample(preds, temperature=temperature)  # Use the sample function to get the next character index
        next_char = index2char[next_index]

        generated_sequence += next_char
        if next_char == ".":
            break
        # move forward with test_chars + next_char
        test_chars = test_chars[1:] + next_char

    return generated_sequence

def sample(predictions, temperature=1.0):
    """
    Sample the next character index based on the model predictions and the temperature parameter.

    Parameters:
    - predictions: The array of predicted probabilities for each character.
    - temperature: Controls the randomness of the predictions (higher values produce more random results).

    Returns:
    - index: The index of the sampled character.
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

# Example usage:
model_path = os.path.join('..', 'pretrained_models', 'lstmTextGeneration.pkl')
char_index_path = os.path.join('..', 'pretrained_models', 'char_index.pkl')
generated_sentence = generate_sentence(model_path, char_index_path, temperature=0.5)
print("Generated Sentence:", generated_sentence)