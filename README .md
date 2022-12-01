
# Disaster Response Pipeline project.
The goal of this project is to build a web application that accepts a distress 
message from an individual and classifies it into one or sevaral categories.


## Significance
The application is to enable those in distress communicate their cry for help thus 
enabling aid agencies easily help them .The categorization helps in knowing the kind 
of help to be offered.
## Dataset
The dataset for this project was obtained from Figure Eight.It comprises of 2 datasets
 that have been merged.The datasets are the Messages dataset which comprises
of the distress messages and the other is the categories dataset which 
contains the classification of the messages.


## Methods used
- Data Visualization
- Machine Learning
- Web Application Development
- Version Control

## Technologies Used
- Python 
- VSCode
- Git 
- HTML
- CSS
- Javascript
- Flask
 
## Project Folder
The project folder comprises of the ML,ETL pipelines and other files that are 
for building the machine learning flask application such as HTML files. The files
in the project folder are described below:

#### ETL Pipeline: processa_data.py

This is where the preparation of the dataset for analysis is done by executing the
following steps:
- Loading the Messages and categories datasets.
- Merging the categories and the Messages dataset to create the df dataframe.
- Data cleaning.
- Storing the data in a SQLite database

#### ML Pipeline:train_classifier.py
This is where the machine learning model for text classification is built and packaged
for production by executing the following steps:

- Loading data from the SQLITE database.
- Building a machine learning pipeline for text processing.
- Model training and tuning using GridSearchCV
- Carrying out predictions using features in the test dataset.
- Exporting the tuned model as a pickle file.

#### Flask Web Application:run.py
This file contains code that creates plots displayed in the front-end of the web application
and also is the entry point of the web application that enables the receipt and classification
of the distress message by the web application possible.






## Instructions
After cloning this github repository:

Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python 
  data/process_data.py data/disaster_messages.csv data/disaster_categories.csv 
  data/DisasterResponse.db

- To run ML pipeline that trains classifier and saves 
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/