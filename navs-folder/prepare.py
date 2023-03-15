import pandas as pd 
import numpy as pd 

import os
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

# -----clean_text---------------

def clean_article(df, col_name):
    cleaned_summaries = []
    for summary in df[col_name]:
        # Normalize the summary text and convert to lowercase
        cleaned_summary = unicodedata.normalize('NFKD', summary)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')\
            .lower()
        cleaned_summary = re.sub(r"[^a-z0-9',\s.]", '', cleaned_summary)
        cleaned_summaries.append(cleaned_summary)
    df[f'cleaned_{col_name}'] = cleaned_summaries

    
    
'''nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
'''

# -----lemmatize_and_Stopwords------------------------------

def lemmatize_text(text):
    """
    Lemmatizes input text using NLTK's WordNetLemmatizer.
    This function first tokenizes the text, removes any non-alphabetic tokens, removes any stop words,
    determines the part of speech of each token, and lemmatizes each token accordingly.
    
    Args:
        text (str): The text to lemmatize.
        
    Returns:
        str: The lemmatized text.
    """
    # Tokenize the text and convert to lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove any non-alphabetic tokens and stop words
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Determine the part of speech of each token and lemmatize accordingly
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag[0].lower()) if pos_tag[0].lower() in ['a', 's', 'r', 'v'] else lemmatizer.lemmatize(token) for token, pos_tag in pos_tags]
    
    # Join the lemmatized tokens back into a string
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text



    
        



    