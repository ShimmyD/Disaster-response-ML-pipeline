import sys
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import numpy as np
import sqlite3
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib


def load_data(database_filepath):
    """create sqlite engine load cleaned dataset from database.
    
    Args: 
        database_name: pass a database name to create engine and load dataset from
    Returns: 
        X: observations
        Y: targets
    """
#     engine = create_engine('sqlite:///'+database_filepath)
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM "+ database_filepath[5:-3], conn)
    print(df.head())
    X =df.message.values
    Y = df.iloc[:,4:].values
    print(X.shape)
    print(Y.shape)
    category_names=df.iloc[:,4:].columns
    return X,Y,category_names

def tokenize(text):
    """create tokenizer for countvectorize.
    
    Args: 
        text: string of raw text message
    Returns: 
        clean_tokens: a list of cleaned words
    """
    stop_words = stopwords.words('english')
    text=text.lower()
    
    #remove punctuations
    text = re.sub(r'[^\w\s]','',text)
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[lemmatizer.lemmatize(token).strip() for token in tokens if token not in stop_words]
    
    return clean_tokens


def build_model():
    """build a pipeline contains steps count vectorization, tfidf transformation,and multi-output classification and gridsearch
    for hyperparameters optimization.
    
    Args: 
        None
    Returns: 
        cv: grid search CV object 
    """
    # set up pipeline steps
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)), ('tfidf',TfidfTransformer()),\
                     ('clf', OneVsRestClassifier(LinearSVC()))])
    
    # specify parameters for grid search
    parameters={'vect__ngram_range': ((1, 1), (1, 2)),'vect__max_df': (0.5, 0.75),\
             'tfidf__use_idf': (True, False),'clf__estimator__loss':['hinge','squared_hinge'],  
           'clf__estimator__C':[1.5,0.8]}
    
    # create grid search object
    cv = GridSearchCV(pipeline,param_grid=parameters,cv=2,verbose=3)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """predict and evaluate model performance.
    
    Args: 
        model: model fitted with X_train
        X_test:test set
        Y_test:test set targets
        category_names: name of all targets
    Returns: 
        classfication_report
    """
    y_pred=model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """pickle model 
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath, ' ',model_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(X_train.shape,' ',Y_train.shape)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()