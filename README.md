#  Identifying The Next Best Seller 
## A Natural Language Processing Analysis Of Books On Goodreads  

### Data Scientist Project Team : Shawn Brown , Manuel Parra, Magdalena Rahn, Brandon Navarrete

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=blue"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=black"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=blue"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=red"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=black"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=blue"></a>

# :star: Goal

Using publically visible data from Goodreads, Wikipedia and Amazon via GitHub, this project aims to acquire, explore and analyse information about books — their popularity, ratings, reviews, keywords, author name, publisher and more – to programmatically determine which factors lead to a book landing on the The New York Times Best Sellers list.



## :star2: Data Overview  

* The data was obtained on 13 and 14 March 2023 using Python coding and the programming utilities BeautifulSoup and Selenium to programmatically acquire data from the public, third-party websites Goodreads, [Wikipedia](https://en.wikipedia.org/wiki/Lists_of_The_New_York_Times_fiction_best_sellers) and Amazon.    

* On GitHub, Maria Antoniak's and Melanie Walsh's [goodreads-scraper](https://github.com/uchidalab/book-dataset) was referenced as initial scaffolding, after which we built our own Python code.    

* Uchidalab's GitHub repository [book-dataset](https://github.com/uchidalab/book-dataset), "Judging a Book by its Cover," _arXiv preprint arXiv:1610.09204 (2016)_, authored by B. K. Iwana, S. T. Raza Rizvi, S. Ahmed, A. Dengel and S. Uchida, was used as a source to obtain random book titles, thus ensuring fair evaluation of book information obtained from other sources.   


    
    
# :star2: Initial Questions

1.  Which words (ngrams ?) appear more often in summaries with a positive sentiment ?  
2.  Which words (ngrams ?) appear more often in summaries with a negative sentiment ? 
3.  Do authors who appear once on the The New York Times Best Sellers list have a higher likelihood of repeat success ?  
4.  Which leads to a higher chance of appearing on the The New York Times Best Sellers list : author + title, or author + publisher ?  



### :dizzy: Project Plan / Process
#### :one:   Data Acquisition

<details> 
    Data was acquired using Python programming and associated libraries and utilities : pandas, NumPy, os, re, time, json, urllib, XPath, BeautifulSoup and Selenium.  
    
    Issues encountered, and resolved, included locating accessible and reliable datasources, applying code across four different computing platforms and learning new data-accessing techniques.

</details>

<details>
<summary> acquire.py </summary>


</details>


#### :two:   Data Preparation

<details>  
    Missing values for book titles were manually imputed, based on the corresponding row's book summary. In cases when the number of pages or year of publication were missing for a given book, the earliest-appearing hardcover book listed on Goodreads was used. Books in languages other than English were dropped, as were duplicates of a given title by the same author and books that only had an audiobook listing on Goodreads.
</details>
    
<details>    
<summary> After manual imputation, the acquired dataframes of random books were all concatenated, and turned into a final dataframe comprising 3998 rows and 11 columns before tidying. The NYT Best Sellers list comprised 1045 rows and 4 columns before tidying.
    
    Tidying included dropping any remaining null values, while deliberately in the collective dataframe keeping NYT Best Seller books that had missing values. The missing values were added in later, manually.
    
    After tidying, the random books dataframe comprised 3686 rows and 19 columns. Columns created included whether the book appeared on the NYT Best Seller list ('bestseller' or 'unsuccessful') and columns to hold normalized title, normalized book summary, lemmatized book summary, and the sentiment score based on the NLTK SentimentIntensityAnalyzer. Additional stopwords were introduced to the stopwords process.
    
    Final columns : title, summary, year_published, author, review_count, number_of_ratings, length, genre, rating, reviews, cleaned_title, cleaned_summary, target, lemmatized_summary, neg, neutral, pos, compound, sentiment.
    </summary>

</details>

        
#### :three:   Exploratory Analysis

<details>
<summary> Questions </summary>

</details>
   
 
#### :four:   Modeling Evaluation

<details>
<summary> Models </summary>
  
</details>


### :medal_sports: Key Findings 
<details>
   
   
<summary> Key Points </summary>
   

</details>


# Recommendation



:electron: # Next Steps


To Reproduce:




## Data Dictionary


|Feature|              Definition|  
| :------|:------|  
|**title**|            title of the book |  
|**summary**|          official Goodreads summary of the book |                                
|**year_published**|   year of publication indicated on the main edition on Goodreads |  
|**author**|           author of the book|  
|**review_count**|     total number of user reviews on Goodreads|   
|**number_of_ratings**|total number of user star ratings on Goodreads|  
|**length**|           length, in pages, of book; if number of pages was missing, the number of pages in the earliest hardcover edition on Goodreads were used|  
|**rating**|           actual star rating from users, with 0 being the lowest and 5 the highest|  
|**reviews**|          text of users' publically available book reviews, when available, up to 10 per book|  
|**cleaned_title**|    book title after normalizing, encoding, decoding and passing through a RegEx statement|  
|**cleaned_summary**|  official Goodreads summary of the book after normalizing, encoding, decoding and passing through a RegEx statement|  
|**target**|           engineered feature indicating whether the book appeared ('bestseller' or 'unsuccessful'), since 1931, on the New York Times Best Seller list|  
|**lemmatized_summary**|lemmatized text of the official Goodreads summary of the book|  
|**neg**|          negative leaning of the sentiment score, based on the official Goodreads summary of the book|  
|**neutral**|     neutral position of the sentiment score, based on the official Goodreads summary of the book|  
|**pos**|          positive leaning of the sentiment score, based on the official Goodreads summary of the book|  
|**compound**|     composite of the negative, neutral and positive sentiment scores, obtained using the NLTK SentimentIntensityAnalyzer|  
|**sentiment**|    word-based indication of the overall sentiment of the official Goodreads summary of the book|   
