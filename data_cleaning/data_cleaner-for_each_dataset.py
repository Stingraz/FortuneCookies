# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 10:44:57 2023

@author: Michelle Fribance

The purpose of this script is to check through each of the 6 fortunes datasets,
clean them of undesirable artifacts, and combine them into a combined dataset.
A spellchecker is applied, punctuation is added where needed, and duplicate 
fortunes are then found and removed, so that the final dataset is a unique set 
of 4675 fortunes.
"""

import pandas as pd
from spellchecker import SpellChecker
import chardet
import os

# Set the display.max_colwidth option to None to show full text in a column
pd.set_option('display.max_colwidth', None)

# Specify the path to the directory in local machine where scripts and datasets are located:
file_path = r"C:\Users\micha\Documents\School_Stuff\IIS\Sem_3\NLP\Project\Datasets"

# Create an empty dataframe to contain all the cleaned datasets:
df_combined = pd.DataFrame(columns=['text'])


####################################################################################
#################### Clean dataset 1: "English_proverbs-1000" ######################
  
# Determine encoding method and import dataset:
file_name = "English_proverbs-1000.csv"
full_file_path = os.path.join(file_path, file_name)
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")
  
df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head()

# Use regex to remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_removed_numbering = df_removed_top_row.copy()
df_removed_numbering.head()


# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
#df_removed_numbering[df_removed_numbering['text'].str.isupper()]
# None found

# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™]'
result = df_removed_numbering[df_removed_numbering['text'].str.contains(pattern)]
print(result)
# None found

# --------------------------------------------------------------------------------- #
# Find any rows containing numbers (often how quotes are represented):
df_removed_numbering[df_removed_numbering['text'].str.contains(r'\d+')]
# none found

# --------------------------------------------------------------------------------- #
# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_removed_numbering], ignore_index=True)
df_combined.head()

# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 0 duplicates
print(f"Number of unique fortunes: {len(df_combined)}")    


####################################################################################
################# Clean dataset 2: "Fortune_cookie_sayings-101" ####################

# Determine encoding method and import dataset:
file_name = "Fortune_cookie_sayings-101.csv"
full_file_path = os.path.join(file_path, file_name)    
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)

# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head()

# Use regex to remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_removed_numbering = df_removed_top_row.copy()
df_removed_numbering.head()


# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
#df_removed_numbering[df_removed_numbering['text'].str.isupper()] 
# None found

# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™]'
result = df_removed_numbering[df_removed_numbering['text'].str.contains(pattern)]
print(result)
# None found

# --------------------------------------------------------------------------------- #

# Find any rows containing numbers (often how quotes are represented):
df_removed_numbering[df_removed_numbering['text'].str.contains(r'\d+')]
# none found

# --------------------------------------------------------------------------------- #

# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_removed_numbering], ignore_index=True)
df_combined.head()

# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 2 duplicates found and removed
print(f"Number of unique fortunes: {len(df_combined)}")    


####################################################################################
################# Clean dataset 3: "Fortunes_by_John_Madison-364" ##################

# Determine encoding method and import dataset:
file_name = "Fortunes_by_John_Madison-364.csv"
full_file_path = os.path.join(file_path, file_name)    
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head()

# Use regex to remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_removed_numbering = df_removed_top_row.copy()
df_removed_numbering.head()

# Use regex to remove bracketed numbers from each fortune (if any): 
# (these indicated the number of times the dataset owner received that fortune)
df_removed_numbering['text'] = df_removed_numbering['text'].str.replace(r'\(\d+\)', '', regex=True)
df_removed_numbering.head()

# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
#df_removed_numbering[df_removed_numbering['text'].str.isupper()] 
# None found

# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™]'
result = df_removed_numbering[df_removed_numbering['text'].str.contains(pattern)]
print(result)
# None found

# --------------------------------------------------------------------------------- #

# Find any rows containing numbers (often how quotes are represented):
df_removed_numbering[df_removed_numbering['text'].str.contains(r'\d+')]
# none found

# --------------------------------------------------------------------------------- #

# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_removed_numbering], ignore_index=True)
df_combined.head()

# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 2 duplicates found and removed
print(f"Number of unique fortunes: {len(df_combined)}")


########################################################################################
################## Clean dataset 4: "proverbs_from_The_Cultureur-52" ###################

# Determine encoding method and import dataset:
file_name = "proverbs_from_The_Cultureur-52.csv"
full_file_path = os.path.join(file_path, file_name)
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head()

# Use regex to remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_removed_numbering = df_removed_top_row.copy()
df_removed_numbering.head()

# Use regex to remove bracketed numbers from each fortune (if any): 
# (these indicated the number of times the dataset owner received that fortune)
df_removed_numbering['text'] = df_removed_numbering['text'].str.replace(r'\(\d+\)', '', regex=True)
df_removed_numbering.head()

# Remove from each row, everything following " |" (eg, | Greek Proverb):
pattern = r' \| '
mask = df_removed_numbering['text'].str.contains(pattern, regex=True)
if mask.sum() > 0:
    print(f"Vertical bar citations found: {df_removed_numbering[mask]}")
    # Create a new DataFrame with only the text up to but excluding the " | " pattern
    new_df = df_removed_numbering['text'].str.extract(r'^(.*?)(?: \| |$)').rename(columns={0: 'text'})
    new_df['text'] = new_df['text'].str.strip()
    # Modify the original dataframe:
    df_removed_numbering['text'] = new_df['text']    
    print(f"\n{len(df_removed_numbering[mask])} citations removed: {df_removed_numbering[mask]}")
else:
    print("No vertical bar citations found.")        
    
# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
df_removed_numbering[df_removed_numbering['text'].str.isupper()]
# None found


# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™]'
result = df_removed_numbering[df_removed_numbering['text'].str.contains(pattern)]
print(result)
# None found

# --------------------------------------------------------------------------------- #

# Find any rows containing numbers (often how quotes are represented):
df_removed_numbering[df_removed_numbering['text'].str.contains(r'\d+')]
# none found

# --------------------------------------------------------------------------------- #

# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_removed_numbering], ignore_index=True)
df_combined.head()

# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 0 duplicates found 
print(f"Number of unique fortunes: {len(df_combined)}")


#####################################################################################
################## Clean dataset 5: "Proverbs_from_Wikipedia-680" ###################

# Determine encoding method and import dataset:
file_name = "Proverbs_from_Wikipedia-680.csv"
full_file_path = os.path.join(file_path, file_name)
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head(10)

# Remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_cleaned = df_removed_top_row.copy()
df_cleaned.head(10)

# Remove the alphabetic section label rows (only present in Wikipedia dataset):
df_cleaned = df_cleaned[~df_cleaned['text'].str.contains(r'\[edit\]')]
# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head(10)

original_number_of_rows = df_cleaned.shape[0]
print(f"Imported dataset has {original_number_of_rows} examples.")


# --------------------------------------------------------------------------------- #
# Remove bracketed numbers from each fortune (if any):
    # Round brackets (1) indicated the number of times the dataset owner received that fortune
    # Square brackets [1] indicated a reference (only found on wikipedia dataset)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\(\d+\)', '', regex=True)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[\d+\]', '', regex=True)
df_cleaned.head(10)
df_cleaned.shape

# Remove square bracketed text, indicating a citation, eg:[a], [citation needed] (only in wikipedia dataset)
mask = df_cleaned['text'].str.contains(r'\[.*\]', regex=True)
print(df_cleaned[mask])
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[.*\]', '', regex=True)
print(df_cleaned[mask])


# --------------------------------------------------------------------------------- #
# Find occurances of the word "proverb" and remove it + the word prior eg, "Arab proverb"
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
# Remove double bracketed text, eg:(... (:::) ...) indicates citations:

pattern = r'\([^()]*\([^()]*\)[^()]*\)'
# Create a boolean mask for rows that match the pattern
mask = df_cleaned['text'].str.contains(pattern, regex=True)
matching_rows = df_cleaned[mask]
print(matching_rows)

# Only 3 rows found. Too complicated to fix, so just remove the rows entirely:
df_cleaned = df_cleaned[~mask]

# Check and make sure no more rows with double bracketed text are found:
mask = df_cleaned['text'].str.contains(pattern, regex=True)
matching_rows = df_cleaned[mask]
print(matching_rows)

# Confirm that no text was removed if it didn't contain a second set of bracketed text:
mask = df_cleaned['text'].str.contains(r'\([^()]*\)', regex=True)
print(df_cleaned[mask])    
    
    
# --------------------------------------------------------------------------------- #
# Remove citations (indicated by " – "):

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

# Return the rows that contain ' – ', followed by some text. This is often how citations are denoted.
dash_pattern = r'[–]'
# Create a boolean mask for rows that match the pattern and get the indices:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
if mask.sum() > 0:
    matching_indices = df_cleaned.index[mask]
    print("\nExamples found containing endash citations:\n")
    print(df_cleaned.iloc[matching_indices])
    
    # Create a new DataFrame with only the text up to but excluding the " - " pattern
    new_df = df_cleaned['text'].str.extract(r'^(.*?)(?:' + dash_pattern + '|$)').rename(columns={0: 'text'})
    new_df['text'] = new_df['text'].str.strip()
    new_df.shape
    #print(new_df.iloc[matching_indices])
    
    # Modify the original dataframe:
    df_cleaned['text'] = new_df['text']
    print("\nExamples with ' - ' citations removed:\n")
    print(df_cleaned.iloc[matching_indices]) 
else:
     print("No citations indicated by an endash were found.")

# --------------------------------------------------------------------------------- #
# Find rows containing an opening bracket followed by a number or "proverb":

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

# Find rows where pattern occurs:
pattern = r'\([^)]*(\d|proverb)[^)]*(?:\)|$)'
df_cleaned[df_cleaned['text'].str.contains(pattern, regex=True)]

df_cleaned_copy = df_cleaned.copy()
# Create a boolean mask for rows that match the pattern
mask = df_cleaned['text'].str.contains(pattern, regex=True)
matching_rows = df_cleaned_copy[mask]
if mask.sum() > 0:
    print("Examples with bracketed text containing numbers or the word 'proverb' found: ", matching_rows)  
    # Remove the specified bracketed text (citations)
    df_cleaned_copy['text'] = df_cleaned_copy['text'].str.replace(pattern, '', regex=True) 
    # Create a boolean mask for rows that have changed
    changed_mask = df_cleaned['text'] != df_cleaned_copy['text'] 
    changed_rows = df_cleaned_copy[changed_mask]
    print("\nExamples with bracketed citations removed: ", changed_rows)
else:
    print("No bracketed text containing numbers or the word 'proverb' found.")

# Copy changes to original df and confirm that pattern was removed:
df_cleaned = df_cleaned_copy
df_cleaned[df_cleaned['text'].str.contains(pattern, regex=True)]


# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
df_cleaned[df_cleaned['text'].str.isupper()]
# None found

# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™]'
result = df_removed_numbering[df_removed_numbering['text'].str.contains(pattern)]
print(result)
# None found

# --------------------------------------------------------------------------------- #
# Find any rows containing numbers (often how quotes are represented):

df_cleaned[df_cleaned['text'].str.contains(r'\d+')]
# Only 1 example found, just remove it:

# Create a boolean mask for rows that contain digits in the 'text' column
mask = df_cleaned['text'].str.contains(r'\d+')

# Remove the rows containing digits
df_cleaned = df_cleaned[~mask]
  
# Check that the rows were removed:
df_cleaned[df_cleaned['text'].str.contains(r'\d+')]


# --------------------------------------------------------------------------------- #
# Check how many examples were removed through cleaning:
examples_removed = original_number_of_rows - df_cleaned.shape[0]
print(f"Cleaned dataset has {df_cleaned.shape[0]} rows remaining.")
print(f"{examples_removed} examples removed through cleaning.")


# --------------------------------------------------------------------------------- #

# Check all examples to see if they have punctuation at the end. If not, add a period:
punctuation_chars = '!,.?' 

# Find indices of examples without punctuation before modification
indices_before = df_cleaned.index[df_cleaned['text'].str[-1].map(lambda x: x not in punctuation_chars)].tolist()

# Print examples without punctuation:
unpunctuated_before = df_cleaned.loc[indices_before]
print(f"\n{len(unpunctuated_before)} examples found without punctuation: \n{unpunctuated_before}")


# Function to add a period if the last character is not punctuation
def add_period(text):
    if pd.notnull(text) and text.strip() and text[-1] not in punctuation_chars:
        return text + '.'
    else:
        return text

# Create a copy of the dataframe and apply the function 
df_cleaned_copy = df_cleaned.copy()
df_cleaned_copy['text'] = df_cleaned_copy['text'].apply(lambda x: add_period(x) if pd.notnull(x) else x)
# Assign the modified copy back to the original dataframe
df_cleaned = df_cleaned_copy

# Print the fixed examples:
unpunctuated_after = df_cleaned.loc[indices_before]
print(f"\n{len(unpunctuated_after)} examples found without punctuation after modification: \n{unpunctuated_after}")


# --------------------------------------------------------------------------------- #


# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_cleaned], ignore_index=True)
df_combined.head()

# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 0 duplicates found 
print("Number of unique fortunes: ", len(df_combined))    
    

####################################################################################
#################### Clean dataset 6: "fortunes-David_Breinl" ######################

# Determine encoding method and import dataset:
file_name = "fortunes-David_Breinl.csv"
full_file_path = os.path.join(file_path, file_name)
with open(full_file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

df = pd.read_csv(full_file_path, header=None, encoding=encoding)
# Add a name to the single column:
df.columns = ['text']
df.head()
df.shape

# Check for blank lines and NULL values and remove:
df_dropped_nulls = df.dropna(how='all')
df_dropped_nulls.shape
# No nulls found

# Remove first row (containing the source website address or in David's dataset, the header "fortunes"):
df_removed_top_row = df_dropped_nulls.drop(0)
df_removed_top_row.head(10)

# Remove the numbers from the beginning of each text:
df_removed_top_row['text'] = df_removed_top_row['text'].str.replace(r'^\d+\.\s*', '', regex=True)
df_cleaned = df_removed_top_row.copy()
df_cleaned.head(10)
# None present for this dataset.

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head(10)

original_number_of_rows = df_cleaned.shape[0]
print(f"Imported dataset has {original_number_of_rows} examples.")


# --------------------------------------------------------------------------------- #
# Remove bracketed numbers from each fortune (if any):
    # Round brackets (1) indicated the number of times the dataset owner received that fortune
    # Square brackets [1] indicated a reference (only found on wikipedia dataset)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\(\d+\)', '', regex=True)
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[\d+\]', '', regex=True)
df_cleaned.shape

# Remove square bracketed text, indicating a citation, eg:[a], [citation needed] (only in wikipedia dataset)
mask = df_cleaned['text'].str.contains(r'\[.*\]', regex=True)
print(df_cleaned[mask])
df_cleaned['text'] = df_cleaned['text'].str.replace(r'\[.*\]', '', regex=True)
print(df_cleaned[mask])
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
print(f"\n{len(result)} examples found in Q&A format: {result}")

mask = df_cleaned['text'].str.contains(pattern)
# Invert the mask using ~ and filter the dataframe
df_cleaned = df_cleaned[~mask]
print(f"\n{len(result)} Q&A examples removed.")

    
# --------------------------------------------------------------------------------- #
# Find any rows which are fully capitalized and make them lowercase, with only first letter capital:
mask = df_cleaned['text'].str.isupper()
matching_rows = df_cleaned[mask]
print(f"\n{len(matching_rows)} examples found in all-caps: {matching_rows}")

# Make selected rows lowercase plus capitalize just the first letter:
df_cleaned.loc[mask, 'text'] = df_cleaned.loc[mask, 'text'].str.lower().str.capitalize()
matching_rows = df_cleaned[mask]
print(f"\nCapitalization fixed for all {len(matching_rows)} examples: {matching_rows}")
# 7 rows fixed


# --------------------------------------------------------------------------------- #
# Find any rows with asterisks or other symbols, and remove them:
pattern = r'[ââ·å¿æ¤°¨ä¹±æÂÃ¢™\*]'
result = df_cleaned[df_cleaned['text'].str.contains(pattern)]
print(result)
# 1 example found; just remove it:

mask = df_cleaned['text'].str.contains(pattern)
# Invert the mask using ~ and filter the dataframe
df_cleaned = df_cleaned[~mask]


# --------------------------------------------------------------------------------- #
# Remove citations (indicated by " --" followed directly by a name, not a space):

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

# Return the rows that contain ' - ', followed by some text. This is often how citations are denoted.
dash_pattern = r' --[a-zA-Z].*'
# Create a boolean mask for rows that match the pattern and get the indices:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print(f"\n{len(matching_rows)} examples found with double dash citations: {matching_rows}")

# Remove the citations:
df_cleaned.loc[mask, 'text'] = df_cleaned.loc[mask, 'text'].str.replace(dash_pattern, '', regex=True)
matching_rows = df_cleaned[mask]
#print(matching_rows)
print(f"\nCitations removed from {len(matching_rows)} examples (but examples were kept in dataframe): {matching_rows}")
# 6 rows fixed


# --------------------------------------------------------------------------------- #
# Remove citations from examples (indicated by " -- " and then a capital letter:
# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

dash_pattern = r' -- [A-Z].*'
# Create a boolean mask for rows that match the pattern:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print(f"\n{len(matching_rows)} examples found with double dash citations: {matching_rows}")

# Remove the citations from the examples:
df_cleaned['text'] = df_cleaned['text'].str.replace(dash_pattern, '', regex=True)
matching_rows = df_cleaned[mask]
print(f"\nCitations removed from {len(matching_rows)} examples (but examples were kept in dataframe): {matching_rows}")
# 138 rows fixed

# Show rows remaining which contain " -- ", but which are not citations:
dash_pattern = r' -- '
# Create a boolean mask for rows that match the pattern:
mask = df_cleaned['text'].str.contains(dash_pattern, regex=True)
matching_rows = df_cleaned[mask]
print(f"\n{len(matching_rows)} examples found with double dashes, which are not citations (left unchanged): {matching_rows}")


# --------------------------------------------------------------------------------- #
# Find rows containing bracketed text which contains a number or "proverb":

# Reset the row numbering:
df_cleaned.reset_index(drop=True, inplace=True)

pattern = r'\(([^()]*(?:\d|proverb)[^()]*)\)'

# Create a boolean mask for rows that match the pattern and get the indices:
mask = df_cleaned['text'].str.contains(pattern, regex=True)
matching_indices = df_cleaned.index[mask]
print("\nExamples found containing numbers or 'proverb':\n")
print(df_cleaned.iloc[matching_indices])
# Leave this example; not a citation.


# --------------------------------------------------------------------------------- #
# Check how many examples were removed through cleaning:
examples_removed = original_number_of_rows - df_cleaned.shape[0]
print(f"Cleaned dataset has {df_cleaned.shape[0]} rows remaining.")
print(f"{examples_removed} examples removed through cleaning.")


# --------------------------------------------------------------------------------- #

# Add the dataframe contents to df_combined:
df_combined = pd.concat([df_combined, df_cleaned], ignore_index=True)
df_combined.shape



####################################################################################
# Now working on entire combined dataframe with all 6 datasets:
    
# --------------------------------------------------------------------------------- #
# Use a spell checker on the dataset to correct any misspelled words:
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
# Check for duplicates and drop them from combined dataframe:  
number_of_duplicates = len(df_combined) - len(df_combined.drop_duplicates())

if number_of_duplicates != 0:
    duplicates = df_combined[df_combined.duplicated()]
    df_combined = df_combined.drop_duplicates()
    print(f"\n{number_of_duplicates} duplicate fortunes removed. Duplicates: ")
    print(duplicates)
else:
    print("\nNo duplicates found.\n")
# 264 duplicates found 
print("Number of unique fortunes: ", len(df_combined))  

# 4675 unique fortunes cleaned and combined.


####################################################################################
# Export the combined dataframe to a csv file
full_path = os.path.join(file_path, 'combined_fortunes-{}.csv'.format(len(df_combined)))
df_combined.to_csv(full_path, encoding='utf-8', index=False)
