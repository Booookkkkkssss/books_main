import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def generate_goodreads_links(df):
    '''
    Uses a list and/or DataFrame of Authors & Titles.

    Scrapes the Goodreads website to extract links to book pages for each book in the input DataFrame.
    
    Uses the book title and author name as query parameters in a search bar to find the book page on Goodreads.
    Stores the extracted links in a new column called 'link' in the DataFrame and also writes them to a text file.
    
    File Writes:
    ------------
    In case the function crashes, jupyter kernel dies, or an unexptected error interrupts the function,
    2 text files are created:

    links.txt
        The actual links saved per line. The last appended links is used as the start page.

    row_index.txt

    Parameters:
    ------------
    df (Pandas DataFrame): The input DataFrame containing information about the books to be searched.

    Returns:
    ------------
    df (Pandas DataFrame): The input DataFrame with a new column called 'link' containing the links to the book pages.
    '''

    # Check if links.txt exists, read the last link as the starter_link
    if os.path.exists("links.txt"):
        with open("links.txt", "r") as file:
            lines = file.readlines()
            starter_link = lines[-1].strip()
    else:
        starter_link = "https://www.goodreads.com/book/show/3450744-nudge"

    # Add new column called 'link' to df
    df['link'] = ""

    driver = webdriver.Chrome()

    # Open a text file for writing
    with open("links.txt", "a") as file:

        # If row_index.txt exists, read the last index and start from the next row
        if os.path.exists("row_index.txt"):
            with open("row_index.txt", "r") as index_file:
                last_index = int(index_file.read().strip())
                start_index = last_index + 1
        else:
            start_index = 0

        for index, row in df.iloc[start_index:].iterrows():
            # loading initial webpage
            driver.get(starter_link)

            # current row content to use in query
            title = row['Title']
            author = row['Author']

            try:
                # add wait for page to finish loading
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="Header"]/div[2]/div[2]/section/form/input[1]')))

                # search GoodReads for "title author"
                search_bar = driver.find_element_by_xpath('//*[@id="Header"]/div[2]/div[2]/section/form/input[1]')
                search_bar.send_keys(title + " " + author)
                search_bar.submit()

                # add wait for page to finish loading
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, 'bookTitle')))

                # extract 1st search result link from page
                links = driver.find_element_by_class_name('bookTitle')
                row_value = links.get_attribute('href')

                # append value to new column called 'link'
                df.at[index, 'link'] = row_value

                # Update starter_link
                starter_link = row_value

                # Write the row_value to the text file
                file.write(f"{row_value}\n")

            except:
                # If no search results or timeout, continue to the next row
                continue

            finally:
                # Save current row index to row_index.txt
                with open("row_index.txt", "w") as index_file:
                    index_file.write(str(index))

    driver.quit()
    return df
def clean_column_text(df, column):
    # Convert column to lowercase and remove text inside parentheses
    df['temp'] = df[column].str.lower().replace(r'\([^()]*\)', '', regex=True)

    # Remove text after colon or hyphen
    df['temp'] = df['temp'].str.split(r'[:\-]').str[0]

    # Remove extra whitespace
    df['temp'] = df['temp'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Modify DataFrame with new column
    new_column = f'cleaned_{column}'
    df[new_column] = df['temp']

    # Remove temporary column
    df = df.drop(columns=['temp'])

    return df
def drop_nulls(df):
    '''
    Drops rows with null values that are less than 1% of the total and columns with null values
    that are greater than 30% of the total.

    Parameters:
    -----------
    df (Pandas DataFrame): The input DataFrame to be processed.

    Returns:
    -----------
    df (Pandas DataFrame): The processed DataFrame.
    '''
    
    # Calculate the percentage of null values in each column
    percent_null = df.isnull().sum() / len(df)

    # Drop columns with null values exceeding 30%
    columns_to_drop = percent_null[percent_null > 0.3].index
    df = df.drop(columns=columns_to_drop)

    # Drop rows with null values less than 1%
    rows_to_drop = df.isnull().sum(axis=1) / len(df.columns) < 0.01
    df = df.loc[rows_to_drop, :]

    return df
def read_csv(file_path):
    '''
    Reads a file from the specified path and returns a Pandas DataFrame.

    Parameters:
    -----------
    file_path (str): The path to the input file.

    Returns:
    -----------
    df (Pandas DataFrame): The contents of the input file as a DataFrame, or None if the file does not exist.
    '''

    # check if file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return None

    # read file and convert to DataFrame
    try:
        df = pd.read_csv(file_path,index_col=0)
        return df

    except Exception as e:
        print(f"Error: Failed to read {file_path} as a DataFrame.")
        print(e)
        return None