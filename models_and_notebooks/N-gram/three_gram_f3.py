import pickle
import random
import os

def generate_cookie(model_path, nb=8):
    def load_model(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            n_grams = model.n_grams
            word_counts = model.word_counts
            sentences_column = model.sentences_column
        return n_grams, word_counts, sentences_column
    
    n_grams, word_counts, sentences_column = load_model(model_path)

    if n_grams is None:
        raise Exception("Model not loaded yet. Need to load the model first")
    
    words = []
    # Choose a random starting sentence
    start_sentence = random.choice(sentences_column.dropna())
    start_word = start_sentence.split()[0]
    next_words = (start_word,)  # Tuple with single word
    words.append(start_word)
    # Generate words until reaching the length or an ending punctuation
    while len(words) < nb:
        if next_words in n_grams:
            # Choose the next word based on the frequency
            next_word = max(n_grams[next_words], key=lambda x: word_counts[x])
            next_words = next_words[1:] + (next_word,)
            words.append(next_word)
            # Check if the next word is an ending punctuation
            if next_word[-1] in ['.', '!', '?']:
                break
        else:
            # If the current n-gram has no following words, choose a n-gram randomly
            next_words = random.choice(list(n_grams.keys()))
            words.extend(list(next_words))

    # Capitalize the first word
    words[0] = words[0].capitalize()
    # Convert all words except the first one to lowercase
    words[1:] = [word.lower() for word in words[1:]]
    # Add a period at the end of the generated sequence if it does not have it
    if words[-1][-1] not in ['.', '!', '?']:
        words[-1] += '.'
    # Replace periods in the middle of the sentence with commas
    for i in range(1, len(words) - 1):
        if words[i] == '.':
            words[i] = ','

    return " ".join(words)

print(generate_cookie(model_path=os.path.join('..', 'pretrained_models', 'n-gram.pkl')))
