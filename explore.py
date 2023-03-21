import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.model_selection

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from scipy import stats
from scipy.stats import norm, binom, pearsonr
from collections import Counter

color1 = '#0e3afa'
color2 = '#68fd7f'  
color3 = '#CD5C5C'

sns.set_style('whitegrid')

seed = 42

#-------------------------------------------------------

def split(df, target):
    train, test = train_test_split(df, test_size=.2, random_state=seed, stratify=df[target])
    return train, test

#-------------------------------------------------------

def get_most_common_negative_ngrams(train, n, top_n):
    '''
        Returns a Pandas DataFrame with the most common n-grams for negative reviews in the given train dataset.
    
        Params:
        --------
            train (Pandas DataFrame): 
                The dataset to extract the negative reviews from.
            n (int): 
                The n-gram size to extract (1 for unigrams, 2 for bigrams, 3 for trigrams).
            top_n (int): 
                The number of rows to return with the most common n-grams.

        Returns:
        --------
            A Pandas DataFrame with two columns: 
                the n-gram words and their frequency count. 
                
            The DataFrame has a total of top_n rows, which are the most 
            common n-grams in the negative reviews. The name of the columns depend on the
            n-gram size.

    '''
    # Subset df split of train into only negative values of 'sentiment' column
    df_neg = train[train['sentiment'].isin(['negative', 'very negative'])]
    
    # Tokenize words
    words = df_neg['cleaned_summary'].str.lower().str.cat(sep=' ').split()

    # Create a list of additional stopwords
    additional_stopwords = ['.']

    # Add the additional stopwords to the default set of stopwords
    stop_words = set(stopwords.words('english') + additional_stopwords)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Get frequency distribution of ngrams
    if n == 1:
        freq_dist = nltk.FreqDist(words)
        most_common = freq_dist.most_common(top_n)
        df_common = pd.DataFrame(most_common, columns=['Words', 'Frequency'])
    elif n == 2:
        freq_dist_bigram = nltk.FreqDist(nltk.bigrams(words))
        most_common = freq_dist_bigram.most_common(top_n)
        df_common = pd.DataFrame(most_common, columns=['Bigrams', 'Frequency'])
    elif n == 3:
        freq_dist_trigram = nltk.FreqDist(nltk.trigrams(words))
        most_common = freq_dist_trigram.most_common(top_n)
        df_common = pd.DataFrame(most_common, columns=['Trigrams', 'Frequency'])
    else:
        print('Invalid value for n')

    return df_common

#-------------------------------------------------------

def explore_question_2_visuals(df):
    
    # create the horizontal count plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.columns[1], y=df.columns[0], data=df.head(15), palette=[color1, color2, color3])
    plt.title(f'Frequency of {df.columns[0]}')
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[0])
    plt.show()

#-------------------------------------------------------

def explore_question_2(train):
    
    top_15_negative_words_df = get_most_common_negative_ngrams(train,1,15)
    top_15_negative_bigrams_df = get_most_common_negative_ngrams(train,2,15)
    top_15_negative_trigrams_df = get_most_common_negative_ngrams(train,3,15)
    
    explore_question_2_visuals(top_15_negative_words_df)
    
    explore_question_2_visuals(top_15_negative_bigrams_df)
    
    explore_question_2_visuals(top_15_negative_trigrams_df)
    
#-------------------------------------------------------

def uni_id_best_seller(train):

    best = train[train['successful'] == True] 
    unsuccessful = train[train['successful'] == False] 

    best_words = pd.Series(' '.join(best.lemmatized_summary).split(' ')).value_counts()
    unsuccessful_words = pd.Series(' '.join(unsuccessful.lemmatized_summary).split(' ')).value_counts()
    all_words = pd.Series(' '.join(train.lemmatized_summary).split(' ')).value_counts()
    
    word_counts = (pd.concat([all_words, best_words, unsuccessful_words], axis=1, sort=True)
                .set_axis(['all', 'best', 'unsuccessful'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    all_words_show = word_counts.sort_values(by='all', ascending=False).head(10)
    
    # figure out the percentage of spam vs ham
    print((word_counts
     .assign(p_unsuccessful = word_counts.unsuccessful / word_counts['all'],
         p_best = word_counts.best / word_counts['all'])
     .sort_values(by='all')
     [['p_unsuccessful', 'p_best']]
     .tail(10)
     .sort_values('p_best')
     .plot.barh(stacked=True),
           
     plt.title('Common Words in Positively Aligned Summaries'),
     plt.xlabel('Probability of Appearing'),
     plt.ylabel('Word'),
     plt.legend(title='Successful Book', loc = 'upper right', labels=['No', 'Yes']))

     )
    
    return best 

#-------------------------------------------------------

def best_bigrams(best):
    
    # this creates a long sentence of all summaries
    best_sent = (' '.join(best.lemmatized_summary))
    
    # this creates a list of bigrams
    bigrams = nltk.ngrams(best_sent.split(), 2)
    
    # count the occurrences of each bigram
    bigram_counts = Counter(bigrams)
    
    # sort the bigrams based on their count
    top_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # plot the top 20 bigrams
    labels, values = zip(*top_bigrams)
    plt.barh(range(len(labels)), values, height=0.9, color=color1)
    plt.title('20 Most Frequently Occuring Bestseller Bigrams(Groups of Two Words)')
    plt.ylabel('Number of Bigram')
    plt.xlabel('Occurrences')
    plt.yticks(range(len(labels)), [' '.join(label) for label in labels])
    plt.savefig('pink_bigrams', dpi=500)
    plt.show()
    
#-------------------------------------------------------

def jointplot_viz(train):
    for col in nums.columns:
        if col == 'number_of_ratings' or col == 'review_count' or col == 'rating':
            for col1 in nums.columns:
                if col1 == 'number_of_ratings' or col1 == 'review_count'or col == 'rating':
                    plt.figure() # create a new figure for each plot
                    sns.jointplot(data=nums, x=col, y= col1, hue= 'target', palette=[color1, color2])
                    plt.title(f"Scatterplot of {col} vs. {col1}")
                    plt.show()

#-------------------------------------------------------

def pearsonr_test(train):
    
    nums = train.select_dtypes(exclude = ['string','object'] )
    
    for col in nums.columns:
        if col != 'target':
            for col1 in nums.columns:
                if col1 != 'target':
                    print(f"pearsonr of {col} vs. {col1}")
            
                    s,p = stats.pearsonr(nums[f'{col}'], nums[f'{col1}'])
                    print(f' stat = {s}, p-value = {p}')
                    print('\n') 

#-------------------------------------------------------

# chi-square function

def chi_sq(a, b):
    '''
    This function will take in two arguments in the form of two discrete variables 
    and runs a chi^2 test to determine if the the two variables are independent of 
    each other and prints the results based on the findings.
    '''
    # Setting an alpha value for the if statement below
    alpha = 0.05
    # Creating the crosstsb of the two features
    result = pd.crosstab(a, b)
    # Running the actual function to determine the chi squared value
    chi2, p, degf, expected = stats.chi2_contingency(result)
    
    print('--------------------------')
    print(f'Chi-square  : {chi2:.4f}') 
    print(f'P-value     : {p:.4f}')
    print('--------------------------')
    if p / 2 > alpha:
        print("We fail to reject the null hypothesis.")
    else:
        print(f'We reject the null hypothesis; there is a relationship between the target variable and the feature examined.')

#-------------------------------------------------------

# plotting all books : success vs page length

def book_len_success(train):
    
    '''
    this function uses the training dataset to plot 
    the target ('successful') against the length in 
    pages of each book. it puts out a barplot.
    '''
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    plt.figure(figsize=(8, 5))

    plt.title('Success Of Book Based On Average Page Length')

    graphed = sns.barplot(x = train['successful'], y = train['length'], palette=[color1, color2]
)

    # set xtick labels and properties
    plt.xticks([0, 1], 
               [ 'Not On List', 'Bestseller'],
               rotation = 25)

    # plt.legend([],[]) --this line unnecessary here
    plt.yticks(np.arange(0, 600, 100))

    # display y axis grids
    # graphed.yaxis.grid(True)

    plt.ylabel('Count')
    plt.xlabel('Appearance On NYT Best Seller List')

    plt.show()

#-------------------------------------------------------

def pearsonr_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a pearsonr test on them and print out 
    the test statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    corr, p = stats.pearsonr(group1, group2)

    alpha = .05
    seed = 42

    print('--------------------------')
    print(f'Correlation = {corr:.4f}') 
    print(f'p-value     = {p:.4f}')
    print('--------------------------')
    print('Is p-value < alpha?', p < alpha)

#-------------------------------------------------------

def sent_vs_len(train):

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=train, x='length', y='compound', hue='successful', palette=[color1, color2])

    plt.title('Comparing Book Length to the Compound Sentiment Score')
    plt.xlabel('Book Length')
    plt.ylabel('Sentiment Score')
    plt.show()

#-------------------------------------------------------



#-------------------------------------------------------



#-------------------------------------------------------
