# Disaster-response-ML-pipeline
Disaster response ML pipeline is a natural language process machine learning pipeline that consists of text processing, feature extraction, modeling and Flask web app display. This ML 
pipeline categorizes communication messages, either direct or through news or social media following a disaster, into 36 categories. The purpose of this Ml pipeline is to help disaster
response organizations filter and pull out the most important and relevant message effciently when a disaster happens. 


## Files
### Raw datasets
* ./data/disaster_categories.csv
 This file contains response categories
* ./data/disaster_messages.csv
 This file conatins raw mesages

### ETL pipeline
 ./data/process_data.py contains ETL pipeline. 
* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Create a SQLite database with an SQLAlchemy engine
* Stores it in a SQLite database
 
### ML pipeline
./models/train_classifier.py creates a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses 
the message to predict classifications for 36 categories (multi-output classification). 
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### Web app
./app/run.py uploads dataset and model to web app. This web app displays dataset visualizations and uses trained model to input text and return classification results.
![](<visualization.JPG>)
![](<classification.JPG>)






