# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 10:44:57 2023

@author: Michelle Fribance

The purpose of this script is to check through each of the 6 fortunes datasets,
clean them of undesirable artifacts, and combine them into a combined dataset.
A spellchecker is applied, punctuation is added where needed, and duplicate 
fortunes are then found and removed, so that the final dataset is a unique set 
of 4632 fortunes.

The first 5 datasets are cleaned using the function clean_dataset(), but David's dataset is
cleaned slightly differently, so is done afterwards, at the bottom of the script.

The combined dataset then has a few augmentations applied.
"""

import pandas as pd
from spellchecker import SpellChecker
import chardet
import os

# Set the display.max_colwidth option to None to show full text in a column
pd.set_option('display.max_colwidth', None)

# Specify the path to the directory in local machine where scripts and datasets are located:
file_path = r"C:\Users\micha\Documents\School_Stuff\IIS\Sem_3\NLP\Project\Datasets"


#################### Define function to clean each dataset ####################

def clean_dataset(file_path, file_name, df_combined):
    full_file_path = os.path.join(file_path, file_name)
    # Determine encoding method and import dataset:
    with open(full_file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")
    
    df = pd.read_csv(full_file_path, header=None, encoding=encoding)
    # Add a name to the single column:
    df.columns = ['text']
    
    # Check for blank lines and NULL values and remove:
    df_dropped_nulls = df.dropna(how='all')
        
    # --------------------------------------------------------------------------------- #
    # Remove first row (containing the source website address or in David's dataset, the header "fortunes"):
    df_removed_top_row = df_dropped_nulls.drop(0)
        
    # --------------------------------------------------------------------------------- #
    # Remove the numbers from the beginning of each fortune (if any):
    df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
    df_cleaned = df_removed_top_row.copy()
    #df_cleaned.head()
    
    # --------------------------------------------------------------------------------- #
    # Remove the alphabetic section label rows (only present in Wikipedia dataset):
    df_cleaned = df_cleaned[~df_cleaned['text'].str.contains(r'\[edit\]')]
    # Reset the row numbering:
    df_cleaned.reset_index(drop=True, inplace=True)
    #df_cleaned.head()

    original_number_of_rows = df_cleaned.shape[0]
    print("\n{} has {} examples.".format(file_name, original_number_of_rows))
    
    # --------------------------------------------------------------------------------- #
    # Remove bracketed numbers from each fortune (if any): 
        # Round brackets (1) indicated the number of times the dataset owner received that fortune
        # Square brackets [1] indicated a reference (only found on wikipedia dataset)
    df_cleaned['text'] = df_cleaned['text'].str.replace(r'\(\d+\)', '', regex=True)
    df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[\d+\]', '', regex=True)
    #df_cleaned.head(10)
    
    # --------------------------------------------------------------------------------- #
    # Remove square bracketed text, indicating a citation, eg:[a], [citation needed] (only in wikipedia dataset)
    mask = df_cleaned['text'].str.contains(r'\[.*\]', regex=True)
    if mask.sum() > 0:
        print("\nSquare bracket citations found: ", df_cleaned[mask])
        df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[.*\]', '', regex=True)
        print("\nCitations removed: ", df_cleaned[mask])
    else:
        print("No citations in square brackets found.")
        
    # --------------------------------------------------------------------------------- #
    # Remove from each row, everything following " | " (eg, | Greek Proverb) - only in Cultureur dataset:
    pattern = r' \| '
    mask = df_cleaned['text'].str.contains(pattern, regex=True)
    if mask.sum() > 0:
        print("Vertical bar citations found: ", df_cleaned[mask])
        # Create a new DataFrame with only the text up to but excluding the " | " pattern
        new_df = df_cleaned['text'].str.extract(r'^(.*?)(?: \| |$)').rename(columns={0: 'text'})
        new_df['text'] = new_df['text'].str.strip()
        # Modify the original dataframe:
        df_cleaned['text'] = new_df['text']    
        print("\n{} citations removed: {}".format(len(df_cleaned[mask]), df_cleaned[mask]))
    else:
        print("No vertical bar citations found.")
        
    # --------------------------------------------------------------------------------- #    
    # Find examples with double bracketed text, eg:(... (:::) ...) indicating citations:
    pattern = r'\([^()]*\([^()]*\)[^()]*\)'
    # Create a boolean mask for rows that match the pattern
    mask = df_cleaned['text'].str.contains(pattern, regex=True)
    matching_rows = df_cleaned[mask]
    if mask.sum() > 0:
        print("Examples with double bracketed text found: ", matching_rows)  # Only print if any matching rows found
        df_cleaned = df_cleaned[~mask] 
        print("{} examples removed which contained citations too difficult to fix.\n".format(len(matching_rows)))
        # Only 3 rows found in Wikipedia dataset only. Too complicated to fix, so just remove the rows entirely.
    else:
        print("No double bracketed text found.")
        
    # --------------------------------------------------------------------------------- #
    # Only the Wiki dataset contains citations indicated by an endash " – " (others are used as regular dashes)
    if file_name == "Proverbs_from_Wikipedia-680.csv":
        # Remove citations (indicated by " – ")   
        dash_pattern = r'[–]'
        # Create a boolean mask for rows that match the pattern and get the indices:
        mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
        matching_indices = df_cleaned.index[mask]
        print("\nExamples found containing endash citations:\n")
        print(df_cleaned.iloc[matching_indices])

        # Create a new DataFrame with only the text up to but excluding the " - " pattern
        new_df = df_cleaned['text'].str.extract(r'^(.*?)(?:' + dash_pattern + '|$)').rename(columns={0: 'text'})
        new_df['text'] = new_df['text'].str.strip()

        # Modify the original dataframe:
        df_cleaned['text'] = new_df['text']
        print("\nExamples with endash citations removed:\n")
        print(df_cleaned.iloc[matching_indices]) 

        # Also only for Wiki: find occurances of the word "proverb" and remove it + the word prior eg, "Arab proverb"
        pattern = r'(?i)\S+\sproverb[s]?'
        # Find occurrences of the word "proverb" and save the indices
        matching_indices_before_removal = df_cleaned[df_cleaned['text'].str.contains(pattern, regex=True, case=False)].index
        print("\nExamples found referencing origin of proverb before removal:\n")
        print(df_cleaned.loc[matching_indices_before_removal])

        df_cleaned['text'] = df_cleaned['text'].str.replace(pattern, '', regex=True).str.strip()
        # Print the rows that were modified
        print("\nExamples found referencing origin of proverb after removal:\n")
        print(df_cleaned.loc[matching_indices_before_removal])

    # --------------------------------------------------------------------------------- #
    # Find rows containing an opening bracket followed by a number or the word "proverb":
    pattern = r'\([^)]*(\d|proverb)[^)]*(?:\)|$)'   
    df_cleaned_copy = df_cleaned.copy()
    # Create a boolean mask for rows that match the pattern
    mask = df_cleaned['text'].str.contains(pattern, regex=True)
    matching_rows = df_cleaned_copy[mask]
    if mask.sum() > 0:
        print("\nExamples with bracketed text containing numbers or the word 'proverb' found: ", matching_rows)  
        # Remove the specified bracketed text (citations)
        df_cleaned_copy['text'] = df_cleaned_copy['text'].str.replace(pattern, '', regex=True) 
        # Create a boolean mask for rows that have changed
        changed_mask = df_cleaned['text'] != df_cleaned_copy['text'] 
        changed_rows = df_cleaned_copy[changed_mask]
        print("\nExamples with bracketed citations removed: ", changed_rows)
    else:
        print("No bracketed text containing numbers or the word 'proverb' found.")
        
    # --------------------------------------------------------------------------------- #    
    # Find any rows containing digits (often how quotes are represented; only presetn in Wiki dataset):
    mask = df_cleaned['text'].str.contains(r'\d+')
    matching_rows = df_cleaned[mask]
    if mask.sum() > 0:
        print("{} examples containing digits found: {}".format(len(matching_rows), matching_rows))
        # Remove the rows containing digits
        df_cleaned = df_cleaned[~mask]   
        print("{} examples removed from dataframe.".format(len(matching_rows)))
    else:
        print("No examples containing digits found.")

    # --------------------------------------------------------------------------------- #        
    # Determine number of examples removed through cleaning process: 
    examples_removed = original_number_of_rows - df_cleaned.shape[0]
    print("Cleaning complete.")
    print("{} examples remaining after cleaning.".format(df_cleaned.shape[0]))
    print("{} examples removed through cleaning.".format(examples_removed))
        
    # --------------------------------------------------------------------------------- #
    # Add the dataframe contents to df_combined:
    df_combined = pd.concat([df_combined, df_cleaned], ignore_index=True)
    print("{} fortunes added to combined dataset.".format(len(df_cleaned)))
    
    return df_combined



#####################################################################################

# Create an empty dataframe to contain all the cleaned datasets:
df_combined = pd.DataFrame(columns=['text'])

# Call function on each dataset:
file_name = "English_proverbs-1000.csv"
df_combined = clean_dataset(file_path, file_name, df_combined)

file_name = "Fortune_cookie_sayings-101.csv"
df_combined = clean_dataset(file_path, file_name, df_combined)

file_name = "Fortunes_by_John_Madison-364.csv"
df_combined = clean_dataset(file_path, file_name, df_combined)

file_name = "proverbs_from_The_Cultureur-52.csv"
df_combined = clean_dataset(file_path, file_name, df_combined)

file_name = "Proverbs_from_Wikipedia-680.csv"
df_combined = clean_dataset(file_path, file_name, df_combined)


####################################################################################
# Clean dataset 6: "fortunes-David_Breinl":
file_name = "fortunes-David_Breinl.csv"
full_file_path = os.path.join(file_path, file_name)

# Determine encoding method and import dataset:
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')

# Remove first row (containing the source website address or in David's dataset, the header "fortunes"):
df_removed_top_row = df_dropped_nulls.drop(0)

# Remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_cleaned = df_removed_top_row.copy()
# None present for this dataset.

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

original_number_of_rows = df_cleaned.shape[0]
print("{} has {} examples.".format(file_name, original_number_of_rows))


# --------------------------------------------------------------------------------- #
# Remove bracketed numbers from each fortune (if any):
    # Round brackets (1) indicated the number of times the dataset owner received that fortune
    # Square brackets [1] indicated a reference (only found on wikipedia dataset)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\(\d+\)', '', regex=True)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[\d+\]', '', regex=True)
df_cleaned.shape

# Remove square bracketed text, indicating a citation, eg:[a], [citation needed]
mask = df_cleaned['text'].str.contains(r'\[.*\]', regex=True)
print("Square bracket citation found in {}: {}".format(file_name, df_cleaned[mask]))
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[.*\]', '', regex=True)
print("\n{} citation removed: {}".format(len(df_cleaned[mask]), df_cleaned[mask]))
# 1 citation removed

# --------------------------------------------------------------------------------- #
# Remove double bracketed text, eg:(... (:::) ...) indicates citations:
pattern = r'\([^()]*\([^()]*\)[^()]*\)'
# Create a boolean mask for rows that match the pattern
mask = df_cleaned['text'].str.contains(pattern, regex=True)
matching_rows = df_cleaned[mask]
if mask.sum() > 0:
    print(matching_rows)  # Only print if any matching rows found
    # None found. 
else:
    print("No double bracketed text found.")

# Confirm that no text was removed if it didn't contain a second set of bracketed text:
mask = df_cleaned['text'].str.contains(r'\([^()]*\)', regex=True)
print("\nExamples containing single bracketed text (not removed): ", df_cleaned[mask])    
    
  
# --------------------------------------------------------------------------------- #
# Remove rows containing Q&A format (these are jokes, not fortunes)
# Find rows containing "Q:" or "Q."
pattern = r'Q:|Q\.'
result = df_cleaned[df_cleaned['text'].str.contains(pattern)]
print("\n{} examples found in Q&A format: {}".format(len(result), result))

mask = df_cleaned['text'].str.contains(pattern)
# Invert the mask using ~ and filter the dataframe
df_cleaned = df_cleaned[~mask]
print("\n{} Q&A examples removed.".format(len(result)))


# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
mask = df_cleaned['text'].str.isupper()
matching_rows = df_cleaned[mask]
print("\n{} examples found in all-caps: {}".format(len(matching_rows), matching_rows))

# Make selected rows lowercase plus capitalize just the first letter:
df_cleaned.loc[mask, 'text'] = df_cleaned.loc[mask, 'text'].str.lower().str.capitalize()
matching_rows = df_cleaned[mask]
print("\nCapitalization fixed for all {} examples: {}".format(len(matching_rows), matching_rows))
# 7 rows fixed


# --------------------------------------------------------------------------------- #
# Remove citations from examples (indicated by " --" followed directly by a name, not a space):

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

dash_pattern = r' --[a-zA-Z].*'
# Create a boolean mask for rows that match the pattern:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print("\n{} examples found with double dash citations: {}".format(len(matching_rows), matching_rows))

# Remove the citations from the examples:
df_cleaned.loc[mask, 'text'] = df_cleaned.loc[mask, 'text'].str.replace(dash_pattern, '', regex=True)
matching_rows = df_cleaned[mask]
print("\nCitations removed from {} examples (but examples were kept in dataframe): {}".format(len(matching_rows), matching_rows))
# 6 rows fixed


# --------------------------------------------------------------------------------- #
# Remove citations from examples (indicated by " -- " and then a capital letter:
# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

dash_pattern = r' -- [A-Z].*'
# Create a boolean mask for rows that match the pattern:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print("\n{} examples found with double dash citations: {}".format(len(matching_rows), matching_rows))

# Remove the citations from the examples:
df_cleaned['text'] = df_cleaned['text'].str.replace(dash_pattern, '', regex=True)
matching_rows = df_cleaned[mask]
print("\nCitations removed from {} examples (but examples were kept in dataframe): {}".format(len(matching_rows), matching_rows))
# 139 rows fixed

# Show rows remaining which contain " -- ", but which are not citations:
dash_pattern = r' -- '
# Create a boolean mask for rows that match the pattern:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print("\n{} examples found with double dashes, which are not citations (left unchanged): {}".format(len(matching_rows), matching_rows))


# --------------------------------------------------------------------------------- #
# Check how many examples were removed through cleaning:
examples_removed = original_number_of_rows - df_cleaned.shape[0]
print("Cleaned dataset {} has {} rows remaining.".format(file_name, df_cleaned.shape[0]))
print("{} examples removed through cleaning.".format(examples_removed))


# --------------------------------------------------------------------------------- #

# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_cleaned], ignore_index=True)



####################################################################################
# Now working on entire combined dataframe with all 6 datasets:
    
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™\*]'
mask = df_combined['text'].str.contains(pattern, regex=True)
matching_rows = df_combined[mask]
print("\n{} examples found with undesirable symbols: {}".format(len(matching_rows), matching_rows))
# 2 examples found; just remove the rows entirely:

# Invert the mask using ~ and filter the dataframe
df_combined = df_combined[~mask]
print("\nAll {} examples with undesirable symbols removed from dataframe.".format(len(matching_rows)))


# --------------------------------------------------------------------------------- #
# Use a spell checker on the dataframe to correct any misspelled words:
# Initialize SpellChecker
spell = SpellChecker()
punctuation_chars = '!,.?' 
def spell_check(text):
    # Remove punctuation from the text (otherwise it thinks every last word in a sentence is misspelled)
    translator = str.maketrans("", "", punctuation_chars)
    text_without_punctuation = text.translate(translator)
    
    words = text_without_punctuation.split()
    corrected_words = [spell.correction(word) if spell.correction(word) != word else word for word in words]
    corrected_count = sum(1 for corrected_word, original_word in zip(corrected_words, words) if corrected_word != original_word)
    corrected_text = ' '.join(word for word in corrected_words if word is not None) if corrected_count > 0 else text
    return corrected_text, corrected_count, list(zip(words, corrected_words))

# Apply spell-checking to the specified column
df_combined['text'], total_corrected_words, examples = zip(*df_combined['text'].map(spell_check))

# Print the number of corrected words
print("Total number of corrected words: {}".format(sum(total_corrected_words)))

# Print examples of originally misspelled words and their corrections
count = 0
for example in examples:
    for original, corrected in example:
        if original != corrected:
            print(f"Original: \"{original}\" Corrected: \"{corrected}\"")
            count += 1
            if count == 5:  # Print 5 examples and then break
                break
    if count == 5:  # Break outer loop when 5 examples are printed
        break
            

# --------------------------------------------------------------------------------- #
# Check all examples to see if they have punctuation at the end. If not, add a period:
punctuation_chars = '!,.?' 

# Function to add a period if the last character is not punctuation
def add_period(text):
    if pd.notnull(text) and text.strip() and text[-1] not in punctuation_chars:
        return text + '.'
    else:
        return text

# Copy the dataframe to track changes
df_combined_original = df_combined.copy()

# Apply the function to the 'text' column
df_combined['text'] = df_combined['text'].apply(add_period)

# Count the number of rows where a period was added
added_period_count = (df_combined['text'] != df_combined_original['text']).sum()
print("Period was added to {} examples.".format(added_period_count)  )


# --------------------------------------------------------------------------------- #
# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print("\n{} duplicate fortunes removed from combined dataset. Duplicates: ".format(number_of_duplicates))
    print(duplicates)
else:
    print("\nNo duplicates found in combined dataset.\n")
# 395 duplicates found 
print("Number of unique fortunes in combined dataset: ", len(df_combined))  

# 4632 unique fortunes cleaned and combined.


####################################################################################
# Export the combined dataframe to a csv file:
full_path = os.path.join(file_path, 'combined_fortunes-{}.csv'.format(len(df_combined)))
df_combined.to_csv(full_path, encoding='utf-8', index=False)
