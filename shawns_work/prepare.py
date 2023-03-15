import pandas as pd 
import numpy as nd 

import os
import unicodedata
import re
import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#-----pulling_the_data----------------------

def get_data(file):
    '''
    Will pull the current data from the 'almost_there' csv file, and prep it for deeper cleaning.
    '''
    df = pd.read_csv(file, index_col=0)
    df = df.drop_duplicates(subset='title')
    
    save = ['Eleven on Top', 'Winter of the World', 'Nothing to Lose', 'Reflected in You']
    sub = df[df['length'].isna()]
    sub1 = sub[sub['title'].isin(save)]
    df = df.dropna(subset='length')
    df = pd.concat([df, sub1], axis=0)
    
    df = df.dropna(subset='summary')
    df = df.dropna(subset='year_published')
    
    df = df.reset_index()
    df = df.drop(columns=['index', 'book_tag'])
    
    df['summary'] = df['summary'].astype('string')
    df['title'] = df['title'].astype('string')
    df['author'] = df['author'].astype('string')
    df['genre'] = df['genre'].astype('string')
    df['length'] = df['length'].astype('float')

    return df

#-----create_target-------------------------

def creat_tar(df, ser):
    target_list = []
    for index, row in df.iterrows():
        if row['cleaned_title'] in ser.tolist():
            target_list.append('best seller')
        else:
            target_list.append('unsuccessful')

    # Add the 'Target' column to the dataframe
    df['target'] = target_list
    
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
    df[f'cleaned_{col_name}'].astype('string')
    
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
    # Stop words
    extra_stop_words = ['book', 'novel', 'work', 'title', 'character', 
              'fuck', 'asshole', 'bitch', 'cunt', 'dick', 'fucking',
             'fucker', 'pussy', 'fag', 'edition', 'story', 'tale', 'genre']
    
    stop_words = set(stopwords.words('english')) | set(extra_stop_words)
    #intialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
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

#------------sentiment_mapping------

def get_sentiment(compound):
    if compound <= -0.5:
        return 'very negative'
    elif compound < 0:
        return 'negative'
    elif compound >= 0.5:
        return 'very positive'
    elif compound > 0:
        return 'positive'
    else:
        return 'neutral'
    
#------------feature_sentiment_score------
    
def feat_sent(text):
    
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    book_synopsis = str(text)
    
    # get the sentiment scores for the synopsis
    sentiment_scores = analyzer.polarity_scores(book_synopsis)
    return pd.Series(sentiment_scores)
    
    
