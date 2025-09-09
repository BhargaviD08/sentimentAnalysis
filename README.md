This project is an interactive sentiment analysis web application built using Python, Streamlit, and scikit-learn. Users can upload a CSV file or provide a CSV URL 
containing product reviews and ratings. The app trains a logistic regression model to classify reviews as Positive or Negative.
This project demonstrates Machine Learning (ML) and Natural Language Processing (NLP) skills, making it ideal for portfolio showcasing.
Features are: 
Upload CSV or enter a CSV URL with columns: review and rating.
Automatically converts ratings into binary sentiment (Positive ≥ 4, Negative < 4).
Uses TF-IDF vectorization with NLTK stopwords removal.
Train-test split for model evaluation.
Logistic Regression classifier with accuracy display.
Sentiment distribution visualized with a pie chart.
Test your own review interactively.
Handles large datasets (up to 1GB) with optional sampling.

Technologies Used :
Python
Streamlit
 — Web app framework
scikit-learn
 — Machine Learning
NLTK
 — Natural Language Processing
Matplotlib
 — Visualization
Pandas
 — Data handling

Demonstrates NLP preprocessing (TF-IDF, stopwords removal).
Shows ML workflow (train-test split, logistic regression, evaluation).
Interactive Streamlit app for real-world demonstration.
Large dataset handling with sampling for efficiency.
