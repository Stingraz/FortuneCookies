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

def generate_sentence(model_path, char_index_path, seed_word, temperature=1.0, SEQLEN = 20):
    """
    Generate a sentence based on a seed word using the provided model and dictionaries.

    Parameters:
    - model_path: Path to the .pkl file containing the trained RNN model.
    - char_index_path: Path to the .pkl file containing char2index and index2char dictionaries.
    - seed_word: The initial word to start the sentence generation.
    - temperature: Controls the randomness of the predictions (higher values produce more random results).

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

    # Ensure seed word length matches SEQLEN
    seed_word = seed_word.lower().ljust(SEQLEN, ' ')[:SEQLEN]

    generated_sentence = seed_word

    for _ in range(SEQLEN):
        X_test = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(seed_word):
            if ch in char2index:
                X_test[0, i, char2index[ch]] = 1

        preds = model.predict(X_test, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index2char[next_index]

        generated_sentence += next_char

        if next_char == ".":
            break

        seed_word = (seed_word + next_char)[-SEQLEN:]

    return generated_sentence

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
seed_word = "Happiness is"
generated_sentence = generate_sentence(model_path, char_index_path, seed_word, temperature=0.5)
print("Generated Sentence:", generated_sentence)