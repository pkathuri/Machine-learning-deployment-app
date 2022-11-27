import sys
import numpy as np 
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    """A function that retrieves dataframes from their location.

    Args:
        messages_filepath : The path to the messages dataframe.
        categories_filepath : The path to the categories dataframe

    Returns:
        df: A dataframe that has the messages and categories 
        dataframes after they have been merged into 1 dataframe.
    """
      # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',on='id')
    return df 

def clean_data(df):
    """A function that cleans data in order to make it suitable for
       model fitting.

    Args:
        df (DataFrame): A dataframe that contains text information that
        has to be cleaned up for text classification using machine learning.
    """
     # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',n=35,expand=True)
     # select the first row of the categories dataframe
    row = categories.iloc[0]

     # use this row to extract a list of new column names for categories.
     # one way is to apply a lambda function that takes everything 
     # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda i:i[:-2]).tolist()
     # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
          # set each value to be the last character of the string
         categories[column] = categories[column].map(lambda i:i[-1])
         # convert column from string to numeric
         categories[column] = categories[column].astype('int')
     # drop the original categories column from `df`
    df=df.drop('categories',axis=1)
     # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories], axis=1)
     #  Removing inappropriate data
    df.related.replace(2,1,inplace=True) 
     # drop duplicates
    df=df.drop_duplicates()  
    return df

def save_data(df, database_filename):
    """A function that saves a data into a database.

    Args:
        df: A clean dataframe that contains text information which 
        is supposed to be stored in a database.  
        database_filename: A database file where the df has been saved 
        in the database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Message', engine, index=False)
     

def main():
    """A function that runs all the functions above and connects this
       script to the command line.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()