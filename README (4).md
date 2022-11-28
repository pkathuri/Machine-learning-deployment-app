
# Disaster Response Pipeline project.
The goal of this project is to build a web application that accepts a distress 
message from an individual and classifies it into one or sevaral categories.


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