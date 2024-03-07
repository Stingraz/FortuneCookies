# Author: Michelle Fribance

import spacy
import nltk
from nltk.lm import MLE


def create_and_fit_MLE_model(training_fortunes_list, n):
    # https://www.nltk.org/api/nltk.lm.html
    nlp = spacy.load("en_core_web_md")  # Load pre-trained spaCy model with word vectors

    # Tokenize training fortunes and pad with special characters at each sequence's start & end
    train_data = [nlp(fortune) for fortune in training_fortunes_list]
    vocabulary = [token.text for tokens in train_data for token in tokens]

    # Create an NLTK model for the reference data:
    MLE_model = MLE(n)

    # Convert training data into n-grams
    train_ngrams = list(nltk.everygrams(vocabulary, max_len=n))

    # Fit the model
    MLE_model.fit([train_ngrams], vocabulary_text=vocabulary)
    return MLE_model


def calculate_single_perplexity(sentence, model, n):
    # Calculates perplexity for a single fortune using the trained NLTK model
    nlp = spacy.load("en_core_web_md")  # Load pre-trained spaCy model with word vectors

    if sentence is None:
        return float('inf')  # Return infinity for None values
    tokens = nlp(sentence)
    ngrams = list(nltk.everygrams(tuple([token.text for token in tokens]), max_len=n))
    fortune_perplexity = model.perplexity(ngrams)
    return fortune_perplexity


def calculate_dataset_perplexity(training_fortunes_list, df_fortunes, n):
    # Create and fit the MLE model:
    MLE_model = create_and_fit_MLE_model(training_fortunes_list, n)

    # Add a column to df for perplexity values and calculate perplexity for each fortune:
    df_fortunes["perplexity"] = df_fortunes["fortunes"].apply(
        lambda fortune: calculate_single_perplexity(sentence=fortune, model=MLE_model, n=n))

    # Count number of fortunes with "inf" perplexity and filter them out:
    inf_perplexity_count = len(df_fortunes[df_fortunes["perplexity"] == float('inf')])
    print(f"Number of fortunes returning 'inf' for perplexity: {inf_perplexity_count}")

    # Calculate avg perplexity excluding "inf" perplexity values:
    average_perplexity = df_fortunes[df_fortunes["perplexity"] != float('inf')]["perplexity"].mean()
    print(f"\nAverage perplexity for all input fortunes excluding 'inf' perplexity: {average_perplexity}")

    return average_perplexity, df_fortunes["perplexity"]
