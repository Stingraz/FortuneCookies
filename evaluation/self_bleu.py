# Author: David

import numpy as np
import copy
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import pandas as pd

def calculate_selfBleu(sentences):
    bleu_scores = []
    sentences_scores = {}
    
    for index, sentence in enumerate(sentences):
        reference_sentences = copy.deepcopy(sentences)
        hypothesis = word_tokenize(sentence)
        del reference_sentences[index]
        references = [word_tokenize(ref) for ref in reference_sentences]
        
        bleu_score = sentence_bleu(references, hypothesis)
        bleu_scores.append(bleu_score)
        sentences_scores[sentence] = bleu_score
    
    # Convert the sentences and their BLEU scores to a DataFrame
    dataframe = pd.DataFrame(list(sentences_scores.items()), columns=['Sentence', 'BLEU Score'])
    
    # Return the average BLEU score and the DataFrame
    return np.mean(bleu_scores), dataframe[dataframe.columns[-1]]
