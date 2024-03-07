# Author: Jana

import os
import pandas as pd
from evaluation import grammar_check
from evaluation.self_bleu import calculate_selfBleu
from evaluation.similarity_within import calculate_similarity_within_generated_fortunes
from evaluation.similarity_training_set import calculate_similarity_to_training_set
from evaluation.meteor import calculate_dataset_meteor
from evaluation.perplexity import calculate_dataset_perplexity


def read_datasets():
    datasets = {
        'GPT2': pd.read_csv(os.path.join('..', 'datasets', 'GPT2_generated_fortunes_100-modular-early-stopping.csv')),
        'Markov': pd.read_csv(os.path.join('..', 'datasets', 'Markov_generated_fortunes_100.csv')),
        'n-gram': pd.read_csv(os.path.join('..', 'datasets', 'n-gram_generated_fortunes_100.csv')),
        'lstm': pd.read_csv(os.path.join('..', 'datasets', 'lstm_generated_fortunes_100.csv')),
        'rnn': pd.read_csv(os.path.join('..', 'datasets', 'rnn_generated_fortunes_100.csv.csv'))
    }
    return datasets


def calculate_evaluation_metrics():
    with open('../datasets/combined_fortunes-4632.csv', 'r') as file:
        training_fortunes_list = file.readlines()

    datasets = read_datasets()
    avgs = []
    results = {}
    result = []

    for dset in datasets:
        sentences = list(datasets[dset]["fortunes"])

        avg_grammar_mistakes, result = grammar_check.calculate_grammar_mistakes(sentences)
        print("grammar mistakes finished")

        avg_bleu, result['BLEU score'] = calculate_selfBleu(sentences)
        print("BLEU finished")

        avg_meteor, result['Meteor'] = calculate_dataset_meteor(training_fortunes_list, sentences)
        print("meteor finished")

        avg_perplexity, result['Perplexity'] = calculate_dataset_perplexity(training_fortunes_list, datasets[dset], 2)
        print("perplexity finished")

        avg_training_similarity, result[
           'Similarity to average training sentence'] = calculate_similarity_to_training_set(training_fortunes_list, sentences)
        print("similarity to training set finished")

        avg_similarity_within = calculate_similarity_within_generated_fortunes(sentences)
        print("similarity within finished")

        # export averages
        dict_avg = {'avg_grammar_mistakes': avg_grammar_mistakes, 'avg_bleu': avg_bleu, 'avg_meteor': avg_meteor,
                    'avg_perplexity': avg_perplexity, 'avg_training_similarity': avg_training_similarity,
                'avg_similarity_within': avg_similarity_within}

        file_path = os.path.join("..", "results", f"{dset}_average_evaluation.csv")
        pd.DataFrame(dict_avg, index=[0]).to_csv(file_path, sep=',', index=False)

        # export results
        file_path = os.path.join("..", "results", f"{dset}_sentence_evaluation.csv")
        result.to_csv(file_path, sep=",", index=False)
        results[dset] = result

    return results, avgs


calculate_evaluation_metrics()
