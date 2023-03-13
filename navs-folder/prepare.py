import pandas as pd 
import numpy as pd 

import os 
from fuzzywuzzy import fuzz, process
import re

# ----cleaned_words------------------------

def create_clean_words(df, ser):
    
    
    # list of titles you want to cross refrence
    norm_series = []
    
    for s in ser:
        # Convert to lowercase
        s = s.lower()
        # Remove special characters
        s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
        norm_series.append(s)
        
    df['clean_titles'] = norm_series
    
    return df

# my attempt at fuzzy wuzzy 
'''# ----fuzzy_wuzzy------------------------

# Define a function to find the best match between a title and a list of options using fuzzy string matching
# I know I normalized but just to make sure let's use the fuzzywuzzy imports to help make sure we have as many matches as possible


def find_best_match(title, options):
    
    # Use the `extractOne()` function to find the best match
    result = process.extractOne(title, options, scorer=fuzz.token_sort_ratio)
    
    # If a match was found, return the best match and the matching score
    if result is not None:
        title, best_match, score = result
        return best_match, score
    # Otherwise, return default values
    else:
        return "", 0

def create_tar(df, match_series):
    
    # Create an empty list to store the target labels
    target_list = []

    # Iterate over each row in the df dataframe
    for index, row in df.iterrows():
        # Find the best match between the clean_titles and the match_series
        best_match, score = find_best_match(row['clean_titles'], match_series)
        # If the matching score is above a certain threshold (e.g. 80), label the row as successful
        if score >= 50:
            target_list.append('Successful')
        # Otherwise, label the row as unsuccessful
        else:
            target_list.append('Unsuccessful')

    # Add the 'Target' column to the dataframe
    df['Target'] = target_list
    
    return df
    '''

# Creating target
def creat_tar(df, ser):
    target_list = []
    for index, row in df.iterrows():
        if row['clean_titles'] in ser.tolist():
            target_list.append('best seller')
        else:
            target_list.append('unsuccessful')

    # Add the 'Target' column to the dataframe
    df['Target'] = target_list
    
    return df

    


    
        



    