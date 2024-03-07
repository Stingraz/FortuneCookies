# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:24:25 2023
 
@author: Michelle Fribance

The purpose of this script is to train a Markov Chain model on the combined 
fortunes dataset, using the markovify library: https://github.com/jsvine/markovify
then generate a set of unique fortunes for evaluation purposes. A trained model 
is then savedas a pickle file, to be used by the fortune_gui.py program.

If you don't have pickle files yet (the pretrained Markov models), then adjust the 
paths on lines 60 and 64 to run this script and create them.

markovify isn't available through Anaconda; must install using pip on your desired env:
pip install markovify

Key parameters of the markovify library:
state_size (default=2): 
    - Determines the number of words that form the state of the Markov Chain.
    - Larger values generally lead to more coherent but less diverse text.

chain (default=None):
    - If you have a pre-built chain (possibly from a previous run), you can 
      provide it to the model using this parameter.
  eg: chain = markovify.Chain.from_text(" ".join(proverbs), state_size=2)
      text_model = markovify.Text("", chain=chain)

max_overlap_ratio (default=0.7):
    - This parameter controls the maximum allowed ratio of overlapping words 
      between the sentences.

max_overlap_total (default=15):
    - This parameter controls the maximum allowed total number of overlapping 
      words between the sentences.

output_max_words (default=200):
    - Maximum number of words in the generated sentence.

tries (default=10):
    - The number of attempts to make a sentence before failing and retrying.

"""

import os
import markovify
import random
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.lm import MLE
import pickle

# Check if punkt is already downloaded
try:
    # Check if punkt is found
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # If not found, download punkt
    nltk.download('punkt')
    
# ---------------------------- Set model parameters -------------------------- #

# Set markovify model parameters:
state_size = 1

# Set number of rounds for retraining:
num_rounds = 25
print(f"\nNumber of retraining rounds set to {num_rounds}")

# Set a seed value for reproducibility
seed_value = 42
random.seed(seed_value)  # Sets the seed for the Python random number generator
print(f"\nSeed value set: {seed_value}")

num_fortunes_to_generate = 100
tries = 100 # Attemps by the Markov model to generate a fortune before starting over

# Set whether or not to filter dissimilar generated fortunes by cosine similarity:
cosine_sim = "false"  
    
# Set the display.max_colwidth option to None to show full text in a column
pd.set_option('display.max_colwidth', None)



############################ Function definitions #############################

def filter_fortunes_with_cosine_similarity(df_generated_fortunes, original_fortunes):
    """ Removes fortunes with too-low similarity to the training set. For word 
        embeddings using spaCy, we use the pre-trained spaCy model 
        "en_core_web_md" (medium-sized English model). This model includes word vectors, 
        and it should work well for general-purpose applications, including fortunes."""
        
    # Load the model:
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Downloading 'en_core_web_md' model...")
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    
    # Tokenize the training fortunes:
    training_tokens = [nlp(fortune) for fortune in original_fortunes]
    
    # Calculate the average vector of all training fortunes:
    average_training_vector = np.mean([token.vector for tokens in training_tokens for token in tokens], 
                                      axis=0)
    
    # Function to calculate cosine similarity with the average training vector:
    def calculate_cosine_similarity_with_average(text):
        if text is None:
            return 0.0  
        tokens = nlp(text)
        vector = np.mean([token.vector for token in tokens], axis=0)
        similarity = cosine_similarity([average_training_vector], [vector])[0][0]
        return similarity
    
    # Apply the cosine similarity function while handling None values
    df_generated_fortunes["Passes_Threshold"] = df_generated_fortunes["Generated fortunes"].apply(
        lambda x: calculate_cosine_similarity_with_average(x) if x is not None else 0.0)
    
    # Filter out rows with None values in the "Generated fortunes" column
    df_generated_fortunes = df_generated_fortunes.dropna(subset=["Generated fortunes"])
    
    cosine_similarity_threshold = 0.5
    df_generated_fortunes = df_generated_fortunes.copy()
    df_generated_fortunes.loc[:, "Passes_Threshold"] = (df_generated_fortunes["Passes_Threshold"] >= cosine_similarity_threshold)
    
    # Filter out generated fortunes below the threshold:
    filtered_fortunes = df_generated_fortunes[df_generated_fortunes["Passes_Threshold"]]
    
    # Print the removed fortunes:
    #removed_fortunes = df_generated_fortunes[~df_generated_fortunes["Passes_Threshold"]]
    #print(f"\nNumber of unique fortunes removed: {len(removed_fortunes)}")
    #print("Removed fortunes:")
    #print(removed_fortunes[["Generated fortunes", "Passes_Threshold"]])
    
    # Print the remaining filtered fortunes:
    #print(f"\nNumber of unique fortunes in filtered_fortunes: {len(filtered_fortunes)}")
    #print("Remaining filtered fortunes:")
    #print(filtered_fortunes[["Generated fortunes", "Passes_Threshold"]])
    
    # Drop the temporary column
    filtered_fortunes = filtered_fortunes.drop(columns=["Passes_Threshold"])
    
    return filtered_fortunes


def evaluate_generated_fortunes(filtered_fortunes, training_fortunes):
    # Evaluate by calculating perplexity for each fortune
    
    # https://www.nltk.org/api/nltk.lm.html
    
    nlp = spacy.load("en_core_web_md")  # Load pre-trained spaCy model with word vectors
    
    # Tokenize training fortunes and pad with special characters at each sequence's start & end
    train_data = [nlp(sentence) for sentence in training_fortunes]
    vocabulary = [token.text for tokens in train_data for token in tokens]
    
    # Create an NLTK model for the reference data:
    n = 2
    nltk_model = MLE(n)
    
    # Convert training data into n-grams
    train_ngrams = list(nltk.everygrams(vocabulary, max_len=n))
    
    # Fit the model
    nltk_model.fit([train_ngrams], vocabulary_text=vocabulary)
    
    # Define a function to calculate perplexity for a given sentence using the trained NLTK model:
    def calculate_perplexity(sentence, model, n):
        if sentence is None:
            return float('inf')  # Return infinity for None values
        tokens = nlp(sentence)
        ngrams = list(nltk.everygrams(tuple([token.text for token in tokens]), max_len=n))
        return model.perplexity(ngrams)
    
    # Add a new column to the DataFrame to store perplexity values
    filtered_fortunes["Perplexity"] = filtered_fortunes["Generated fortunes"].apply(
        lambda x: calculate_perplexity(x, nltk_model, n))
    
    # Sort the dataframe by perplexity in ascending order (lower perplexity is better):
    filtered_fortunes = filtered_fortunes.sort_values(by="Perplexity", ascending=True)
    
    #print(filtered_fortunes[["Generated fortunes", "Perplexity"]])
    
    # Check for duplicates and remove:  
    number_of_duplicates = len(filtered_fortunes) - len(filtered_fortunes.drop_duplicates())
    
    if number_of_duplicates > 0:
        #duplicates = filtered_fortunes[filtered_fortunes.duplicated()]
        filtered_fortunes = filtered_fortunes.drop_duplicates()
        #print(f"\n{number_of_duplicates} duplicate fortunes removed from filtered_fortunes. Duplicates: ")
        #print(duplicates)
    #else:
        #print("\nNo duplicates found in filtered_fortunes.\n") 
        
    # Print the top fortunes:
    #top_n = 10
    #print(f"\nTop {top_n} generated fortunes (based on lower perplexity being better):")
    #print(filtered_fortunes.head(top_n)[["Generated fortunes", "Perplexity"]])
    
    # Filter out rows with "inf" perplexity:
    valid_perplexity_df = filtered_fortunes[filtered_fortunes["Perplexity"] != float('inf')]
    
    return valid_perplexity_df

    
def generate_fortune(state_size):
    # Load the trained Markov model from the saved file
    with open(os.path.join("pretrained_models", f"trained_markov_model-state_size_{state_size}.pkl"), "rb") as f:
        text_model = pickle.load(f)
        
    # Generate a single fortune
    try:
        print("Generating fortune...")
        generated_fortune = text_model.make_sentence(
            max_words=15, max_overlap_ratio=0.5, tries=100)
        return generated_fortune
    except Exception as e:
        print(f"Error generating fortune: {e}")
        return None
     
        
######################## End of Function Definitions ##########################

def main():
    # -------------------------------- Load data -------------------------------- #
    
    # Open the original combined_fortunes dataset
    training_fortunes_path = os.path.join("datasets", 'combined_fortunes-4632.csv')
    
    # Set the original dataset as the training fortunes list for the first round
    with open(training_fortunes_path, 'r') as file:
        training_fortunes = file.readlines()
    
    
    previous_fortunes = training_fortunes # Define this here for later
    
    
    # ----------------- Build the initial Markov Chain model -------------------- #
    
    # Combine the fortunes into a single string for Markovify
    text_model = markovify.Text(" ".join(training_fortunes), state_size=state_size)
    print(f"\nMarkov model built using state size {state_size}")
    
    # Initialize dataframe
    df_generated_fortunes = None
    
    
    ##################### Retrain the model in several rounds #####################
    
    for i in range(1, num_rounds + 1):
        
        # Generate fortunes and save to a list:
        print("Generating fortunes...")
        generated_fortunes = [text_model.make_sentence(max_words=15, max_overlap_ratio=0.5, 
                                tries=tries) for _ in range(num_fortunes_to_generate)]
        
        # If df already exists (ie, all retraining rounds), then just overwrite that df:
        if df_generated_fortunes is not None and "Generated fortunes" in df_generated_fortunes.columns:
            # Check if the length of the generated fortunes list matches the length of the DataFrame
            if len(generated_fortunes) != len(df_generated_fortunes):
                # Adjust the size of the generated fortunes list to match the length of the DataFrame
                generated_fortunes = generated_fortunes[:len(df_generated_fortunes)]        
            # Overwrite the existing column with the new generated fortunes
            df_generated_fortunes["Generated fortunes"] = generated_fortunes
        else:
            # If df_generated_fortunes doesn't exist yet or doesn't have the column,
            # create a new DataFrame with the generated fortunes column
            df_generated_fortunes = pd.DataFrame({"Generated fortunes": generated_fortunes})
       
        # Remove blank fortunes:
        df_generated_fortunes = df_generated_fortunes.dropna(how='all')
        
        
        # ----------- Check for failed or duplicate generated fortunes -------------- #
        
        # Print the count of null values (number of failed attempts of the Markov model to generate a fortune):
        print(f"\nFailed attempts to generate a fortune: {df_generated_fortunes.isnull().sum()['Generated fortunes']}/{num_fortunes_to_generate}")
        
        # Remove blank lines (failed attempts at generating a fortune):
        df_generated_fortunes = df_generated_fortunes.dropna(how='all')
        
        # Check for duplicates and remove:  
        number_of_duplicates = len(df_generated_fortunes) - len(df_generated_fortunes.drop_duplicates())
        
        if number_of_duplicates > 0:
            #duplicates = df_generated_fortunes[df_generated_fortunes.duplicated()]
            df_generated_fortunes = df_generated_fortunes.drop_duplicates()
            #print(f"\n{number_of_duplicates} duplicate fortunes removed from df_generated_fortunes. Duplicates: ")
            #print(duplicates)
        #else:
            #print("\nNo duplicates found in df_generated_fortunes.\n")
        
        #print(f"\nNumber of unique fortunes in df_generated_fortunes: {len(df_generated_fortunes)}") 
        
        # Print several successful fortunes for manual inspection:
        #print(df_generated_fortunes[:5])
        
        
        # Filter bad fortunes using Cosine Similarity / Word Embeddings:
        if cosine_sim == "true":
            # Filter out fortunes below the threshold
            filtered_fortunes = filter_fortunes_with_cosine_similarity(df_generated_fortunes, previous_fortunes)
        else:
            filtered_fortunes = df_generated_fortunes
        
        
        # -------- Evaluate generated fortunes by calculating Perplexity ------- #
        
        valid_perplexity_df = evaluate_generated_fortunes(filtered_fortunes, training_fortunes=previous_fortunes)
        
        
        # Calculate the average perplexity for all generated sentences:
        average_perplexity = valid_perplexity_df["Perplexity"].mean()
        print(f"\nRound {i}: Average Perplexity for all generated fortunes: {average_perplexity}")
        
        # Extract the "Generated fortunes" column and convert it to a list:
        generated_fortunes_list = valid_perplexity_df["Generated fortunes"].tolist()
        
        
        # ------- Combine top generated fortunes with original dataset ---------- #
        
        # Calculate the number of items to keep (20% of the length of generated_fortunes_list)
        num_to_keep = int(len(generated_fortunes_list) * 0.2)
        
        # Keep the first num_to_keep items
        expanded_fortunes = previous_fortunes + generated_fortunes_list[:num_to_keep]
        
        previous_fortunes = expanded_fortunes  # Define this here for the next round of training
        
        
        # ------------ Retrain the model on the expanded dataset ---------------- #
    
        # Join the expanded fortunes with a space and retrain the model
        text_model = markovify.Text(" ".join(expanded_fortunes), state_size=state_size)
      
    
    
    # Save the final text_model to be used by the fortune_gui.py program:
    with open(os.path.join("pretrained_models",f"trained_markov_model-state_size_{state_size}.pkl"), "wb") as f:
        pickle.dump(text_model, f)
    
    
    """############################## Export Datasets ################################
    
    # Check for or create "Generated_Fortunes" folder, to store results:
    generated_fortunes_folder = os.path.join(Markov_Chains_folder_path, "Generated_Fortunes")
    if not os.path.exists(generated_fortunes_folder):
        os.makedirs(generated_fortunes_folder)
        
        
    # --------- Export generated fortunes to CSV for manual evaluation ---------- #
    
    # Export the generated DataFrame to a CSV file:
    csv_file_path = os.path.join(generated_fortunes_folder, 
                                 f"generated_fortunes-rounds_{num_rounds}.csv")
    valid_perplexity_df.to_csv(csv_file_path, index=False)
    print(f"\nGenerated fortunes dataset exported to: {csv_file_path}")  
    
    
    # ----- Export expanded fortunes to CSV for maybe use by other models ------ #
    
    # Create DataFrame out of original dataset and generated fortunes:
    expanded_fortunes_df = pd.DataFrame({"Generated fortunes": expanded_fortunes})
    
    # Export the expanded DataFrame to a CSV file for final use by the Gui:
    csv_file_path = os.path.join(generated_fortunes_folder, 
                                 f"expanded_fortunes-rounds_{num_rounds}.csv")
    
    expanded_fortunes_df.to_csv(csv_file_path, index=False)
    print(f"\nExpanded fortunes dataset exported to: {csv_file_path}") """
    
    
if __name__ == "__main__":
    main()
    