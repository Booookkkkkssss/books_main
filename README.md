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

1.  Which words (ngrams?) appear more often in summaries with a positive sentiment?  
2.  Which words (ngrams?) appear more often in summaries with a negative sentiment?  
3.  Do authors who appear once on the The New York Times Best Sellers list have a higher likelihood of repeat success?   
4.  Which combination of features led a higher chance of appearing on the NYT Best Seller list?   



### :dizzy: Project Plan / Process
#### :one:   Data Acquisition

<details> <summary> Acquisition </summary>
Data was acquired using Python programming and associated libraries and utilities : pandas, NumPy, os, re, time, json, urllib, XPath, BeautifulSoup and Selenium.    
    
Issues encountered, and resolved, included locating accessible and reliable datasources, applying code across four different computing platforms, learning new data-accessing techniques and website obstacles.

</details>

<details>
<summary> acquire.py </summary>


</details>


#### :two:   Data Preparation

<details>  
    <summary> 
Missing values for book titles were manually imputed, based on the corresponding row's book summary. In cases when the number of pages or year of publication were missing for a given book, the earliest-appearing hardcover book listed on Goodreads was used. Books in languages other than English were dropped, as were duplicates of a given title by the same author and books that only had an audiobook listing on Goodreads. Genres with less than 8 titles were dropped, as were picture books.  
    </summary>
</details>
    
<details>    
After manual imputation, the acquired dataframes of random books were all concatenated, and turned into a final dataframe comprising 3998 rows and 11 columns before tidying. The NYT Best Sellers list comprised 1045 rows and 4 columns before tidying.   
    
Tidying included dropping any remaining null values, while deliberately in the collective dataframe keeping NYT Best Seller books that had missing values. The missing values were added in later, manually.  
    
After tidying, the random books dataframe comprised 3665 rows and 19 columns. Columns created included whether the book appeared on the NYT Best Seller list ('bestseller' or 'unsuccessful') and columns to hold normalized title, normalized book summary, lemmatized book summary, and the sentiment score based on the NLTK SentimentIntensityAnalyzer. Additional stopwords were introduced to the stopwords process.  
    
Final columns : title, summary, year_published, author, review_count, number_of_ratings, length, genre, rating, reviews, cleaned_title, cleaned_summary, target, lemmatized_summary, neg, neutral, pos, compound, sentiment.  
   
</details>

        
#### :three:   Exploratory Analysis


<details>
<summary> Questions initially identified during discussion sessions were refined during exploration. Some of the inital questions were answered, while others, which demanded asking after increased familiarity with the data, were explored and responded to.   
    </summary>

<details>
Question 1: Looking at bigrams, best-selling author names appeared often, as did character names from series (possibly due to it being a small sample in the data set or people being drawn to series due to emotional connection to characters) and place names.    

Question 4: Which combination of features led a higher chance of appearing on the NYT Best Seller list ? The greater the number of reviews and the greater the number star ratings correspond to a higher overall rating.  A slight correlation was found between  having a higher negative summary sentiment score and being a bestseller.   
    
Other questions:  
    a. How many are books successful and not successful? 0.48% were found to be successful in our dataset.  
    b. Which authors are having the most success? J.D. Robb, Stephen King and Rick Riordan topped the list.  
    c. The max rating for bestseller books is 4.76, while the average rating for bestsellers was 4.10. In unsuccessful books, the average score was 4.00, but the max rating was 4.8.
    d. What was the distribution of summary sentiment scores based on review count?    
        For bestsellers, books with a very positive sentiment score had the highest number of reviews, followed by books with a positive sentiment score.  
        For non-bestsellers, books with a negative summary-sentiment score had the highest number of reviews, followed by books with a very negative or a very positive sentiment score.  
        For the overall train dataset, books with a negative summary-sentiment score had the highest number of reviews, followed by books with a positive sentiment score.  
        Of the bestseller sentiment scores in the train dataset, 65 had very negative scores, 7 had negative, 1 had neutral, 11 had positive and 43 had very positive.  
    
    e. Does the length of a book have a relationship to its successs?   
        The mean length of bestsellers was 477 pages, the median was 400 pages. The standard deviation was about 205 pages. So, 68% of NYT bestsellers had a length of 272 to 682 pages.  
        Non-bestsellers had an average length of about 355 pages, with a standard deviation of about 175 pages. So, 68% of non-bestsellers had a length between 180 and 530 pages.  
        Using the CDF (cumulative density function) based on the low end of the non-bestseller standard deviation, it was found that there was a 7pc chance of a successful book having a length of 180 pages or less.  
        
    
    f. Of all authors, which ones had the most books published ?  
        J.D. Robb, Stephen King and Louise Penny were the most prolific.  
    
    g. Which genres are most prevalent? Fiction, non-fiction, fantasy and romance titles topped the list.  
    
    h. What is the relationship between the sumamry sentiment score and the book length?  
        There was a weak negative correlation, as demonstrated by the Pearson's R test.
    
    i. Is there a relationship between length and year published?   
        Data was plotted and Chi-square test were run on bestsellers, non-bestsellers and on the full train dataset.  
        On the train dataset, there was a strong positive correlation between length and year published.   
        For bestsellers, the null hypothesis (there is no relationship between lenght and year published) could NOT be rejected.  
        Non-bestsellers showed a strong positive correlation between length and year published.  
 
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
