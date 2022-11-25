import sys
import numpy as np
import pandas as pd
import nltk
import sqlalchemy
import string
from sqlalchemy import create_engine
import sqlite3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    # load data from database
     engine = create_engine('sqlite:///'+database_filepath)
     df =  pd.read_sql_query("SELECT * from Message", engine)
     X = df['message'].values
     y = df.iloc[:, 4:].values
     category_names = df.iloc[:, 4:].columns.values  
     return X,y,category_names

def tokenize(text):
    s =  set(string.punctuation)  
   
    stop_words=set(stopwords.words('english'))
   
    word_tokens = word_tokenize(text.lower())
    filtered_word = []
    for i in word_tokens:
        if i not in s:
            filtered_word.append(i)

    filtered_sentence = []
    for w in filtered_word:
        if w not in stop_words:
            filtered_sentence.append(w)
   
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in filtered_sentence:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # parameters={"clf__estimator__criterion":['gini','entropy'],
    #         "clf__estimator__max_features":['auto','sqrt','log2'],
    #          "clf__estimator__bootstrap":[True,False]}
    # cv = GridSearchCV(pipeline,param_grid=parameters,verbose=0,cv=2)
    return   Pipeline(
             [('cv', CountVectorizer(tokenizer=tokenize)),
             ('tfidf', TfidfTransformer()),
             ('clf',MultiOutputClassifier(RandomForestClassifier(bootstrap=True,criterion='entropy',max_features='log2')))]   
             )


def evaluate_model(model, X_test, y_test, category_names):
     y_pred=model.predict(X_test)     
     print(classification_report(y_test, y_pred))

def save_model(model, model_filepath):
     with open(model_filepath, 'wb') as best:
         pickle.dump(model, best, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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