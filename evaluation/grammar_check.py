# Author: Jana

import numpy as np
import pandas as pd
import language_tool_python


def calculate_grammar_mistakes(sentences):
    grammar_mistakes = []
    sentences_scores = {}

    tool = language_tool_python.LanguageToolPublicAPI('en-US')

    for i in sentences:
        mistakes = len(tool.check(i))
        grammar_mistakes.append(mistakes)
        sentences_scores[i] = mistakes
    dataframe = pd.DataFrame(list(sentences_scores.items()), columns=['Sentence', 'Grammar mistakes'])

    return np.mean(grammar_mistakes), dataframe
