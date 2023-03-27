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

#---------------------------------------------

def prep_data(filename):
    
    df = get_data(filename)
    
    df.drop(columns='index')
    
    clean_article(df, 'title')
    clean_article(df, 'summary')
    clean_article(df, 'reviews')
    
    df1 = pd.read_csv('fiction-and-non-fiction-top-best-sellers.csv', index_col=0)
    
    clean_article(df1, 'Book')
    ser = df1['cleaned_Book']
    
    creat_tar(df, ser)
    
    # convert 'year' column to string type using .astype()
    df['year'] = df['year'].astype(str)
    # extract the year from the 'year' column using the lambda function
    df['year'] = df['year'].apply(lambda x: x[-4:])
    
    # use regex on col to isolate page length
    df['length'] = [re.findall(r'\d', str(x)) for x in df['length']]
    # join the page numbers together
    df["length"]= df["length"].str.join("")
    
    df.rating = df['year'].astype(str)
    # remove the last 8 characters from each string and return the result
    df.rating_count = [s[:-8] for s in df.rating_count]
    
    # drop duplicate title-author combos
    df.drop_duplicates(subset = ['title', 'author'], inplace = True)
    
    df.publisher = df.publisher.astype(str)
    df.publisher = [s.split('by')[1].strip() if len(s.split('by')) > 1 else '' for s in df.publisher]
    
    genre_counts = df['genre'].value_counts()
    genres_to_remove = genre_counts[genre_counts < 8].index
    # remove the rows with those genres "filtering"
    df = df[~df['genre'].isin(genres_to_remove)]
    
    df = df[df['genre'] != 'Picture Books']
    
    df['lemmatized_summary'] = df['cleaned_summary'].apply(lemmatize_text)
    df[['neg', 'neutral', 'pos', 'compound']] = df['summary'].apply(feat_sent)
    df['sentiment'] = df['compound'].apply(get_sentiment)
    
    return df

#-----pulling_the_data----------------------

def get_data(file):
    '''
    Will pull the current data from the 'almost_there' csv file, and prep it for deeper cleaning.
    '''
    # original df
    df = pd.read_csv('weekend_dataset.csv', index_col=0)    
    # combining review cols
    df['reviews'] = df['Review 1'] + ' ' + df['Review 2'] + ' ' + df['Review 3'] + ' ' + df['Review 4'] + ' ' +  df['Review 5']
    # dropping unneeded cols
    df.drop(['Review 1', 'Review 2', 'Review 3', 'Review 4', 'Review 5'], axis = 1, inplace = True)
    # rename columns
    df = df.rename(columns={'Book Name':'title','Synopsis':'summary', 'Link':'link', 'page_count':'length'})
    
    df = df.reset_index()
                                               
    return df

#-----create_target-------------------------

def creat_tar(df, ser):
    target_list = []
    for index, row in df.iterrows():
        if row['cleaned_title'] in ser.tolist():
            target_list.append(1)
        else:
            target_list.append(0)

    # Add the 'Target' column to the dataframe
    df['successful'] = target_list
    df['successful'] = df['successful'].astype(bool)

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
              'fucker', 'pussy', 'fag', 'edition', 'story', 'tale', 
              'genre', 'new york times', 'ny times', 'nyt', 'new', 'york',
              'times', 'bestseller', 'author', 'bestselling', 'one', 'two']
    
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
    
    
