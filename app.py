# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk

# -----------------------------
# Streamlit settings
# -----------------------------
st.set_page_config(page_title="Flipkart/Amazon Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis on Flipkart / Amazon Reviews")

# Download stopwords
nltk.download('stopwords')
english_stopwords = stopwords.words("english")

# -----------------------------
# File upload or URL input
# -----------------------------
st.subheader("Upload Dataset or Enter URL")
uploaded_file = st.file_uploader("Upload CSV file with columns: 'review', 'rating'", type=["csv"])
url_input = st.text_input("Or enter a CSV URL:")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif url_input.strip() != "":
    try:
        df = pd.read_csv(url_input)
    except Exception as e:
        st.error(f"Error reading URL: {e}")

# -----------------------------
# Dataset preview and validation
# -----------------------------
if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Validate columns
    if "review" not in df.columns or "rating" not in df.columns:
        st.error("CSV must contain 'review' and 'rating' columns.")
    else:
        # Convert ratings into sentiment (1 = Positive, 0 = Negative)
        df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

        # Optional: sample large datasets
        if len(df) > 100000:
            st.warning("Dataset is very large. Sampling 100k rows for faster processing.")
            df = df.sample(100000, random_state=42)

        # -----------------------------
        # Train-test split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            df["review"], df["sentiment"], test_size=0.2, random_state=42
        )

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words=english_stopwords, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully with Accuracy: {acc:.2f}")

        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=["Positive", "Negative"], autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

        # -----------------------------
        # Test custom review
        # -----------------------------
        st.subheader("Test Your Own Review")
        user_review = st.text_area("Enter a review here:")

        if st.button("Predict Sentiment"):
            if user_review.strip():
                user_vec = vectorizer.transform([user_review])
                prediction = model.predict(user_vec)[0]
                label = "Positive 😊" if prediction == 1 else "Negative 😡"
                st.success(f"Prediction: {label}")
            else:
                st.warning("Please enter some text.")
