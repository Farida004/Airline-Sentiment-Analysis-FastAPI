# Airline-Sentiment-Analysis-FastAPI

The following project is about airline sentiment analysis on a given airline-sentiment-analysis.csv data. Project comprises of several steps such as analyzing the data, cleaning the data,
providing the visualizations, building several models and develop an API with FastAPI.

The task was:

* Build a binary classification model using any library of your choice (Random Forest Classifier)
* Use OOPs concept to train the model, reading data for training, and implement inference class.
* Develop an API server on python using Fast API.
* Implement APIs that can accept an english text and respond with the predicted sentiment.
* Upload the entire code to a newly created Git Repo.
* Integrate Swagger documentation for your Rest API endpoint. 

In the repo you can find OOP.py and imports.py files which consist of OOP implementation of data preparation and model creation functions.sentiment.py is the main file to run for FastAPI server. 2 pkl files vectorized and rfc-model are trained countvectorizer and randomforest model used for API.  Additionally you can find a Report which describes the classification task in details. 

Link to the [video](https://www.youtube.com/watch?v=OTSId1nzdvo)

# How to run:

* uvicorn sentiment:app --reload

# Endpoint
Project contains Swagger UI:
* http://127.0.0.1:8000/docs

HTTP requests can be sent to:
* http://127.0.0.1:8000/sentiment_analysis/?text="sometext"
