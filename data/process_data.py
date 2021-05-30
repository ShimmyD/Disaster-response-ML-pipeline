import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """Method to load messages and categories datasets and merge together.
    
    Args: 
        None
    Returns: 
        merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner',on='id')
    return df

def clean_data(df):
    """Method to clean dataset
    
    Args: 
        raw dataset
    Returns: 
        cleaned dataset for ML pipeline
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-').str[0].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #iterate through each column to get 1 or 0 as lable for each category and covert the column to numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace 2 in related category to 1 as each category can only have binary values 0 or 1. 

    # replace 2 in related category to 1 as each category can only have numbers 0 or 1. 

    categories=categories.replace(2,1)
    
    # drop original categories column from df
    df.drop(columns=['categories'],inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe 
    df = pd.concat([df,categories],axis=1)
    
    #drop duplicates
    df=df[~df.message.duplicated()]
    return df


def save_data(df, database_filename):
    """create sqlite engine and save dataset into database.

        Args: 
            df: dataset to be saved into database
            database_name: pass a database name to create engine and sqlite3.connect
        Returns: 
            None
      """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename[5:-3], engine, index=False,if_exists='replace')
 
 
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print(database_filepath)
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
