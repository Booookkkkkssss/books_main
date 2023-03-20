import requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup 
import time
import sketch

import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------

def get_book_urls(filename, rng=15):
    '''
    This function will take in an optional range and scrape and then format the url of a
    books page to be input into a file to use for further scraping. Needs additional work to get it to work in a more universal way, 
    but functions as is. Need to edit function url for each different web page currently.
    '''
    books = []
    # initial loop to gather the data from Goodreads
    for i in range(1, rng):
        # The url where the books are gathered. This is the part that needs tweaking so that the function can just take in a url. 
        # Need to research how to format the string properly.
        url = f'https://www.goodreads.com/list/show/264.Books_That_Everyone_Should_Read_At_Least_Once?page={i}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')  
        page = soup.find_all('a', class_='bookTitle')
        # Checking the status code as function runs
        print(response.status_code)
        # Appending the gathered data into a list of lists
        for i in range(0, len(page)):
            # Referencing the specific section of the hmtl to find the url
            href = page[i]['href']
            # Appending the url onto the list after shaving off the unneeded part with Regex   
            books.append(re.findall(r'\d.*', href))
            # books.append(href)
        # A short sleep timer so Goodreads doesn't get upset
        time.sleep(3)        
    books_list = []
    #url = 'https://www.goodreads.com/book/show/' + book_id
    # Short loop to turn the above created list of lists into a list for better formatting
    for i in range(len(books)):
        book = books[i][0]
        books_list.append(book)
    # Creating/Updating a file formatted properly for use later    
    with open(filename, "w") as output:
    # output.write(str(books_list))
        for i in books_list:
            output.write(i + '\n')
            
#-----------------------------------------------------



#-----------------------------------------------------



#-----------------------------------------------------



#-----------------------------------------------------



#-----------------------------------------------------
