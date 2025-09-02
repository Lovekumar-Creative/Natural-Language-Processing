# app.py
import streamlit as st
import nltk
import re
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Download required NLTK resources (if not already done)
nltk.download('stopwords')
nltk.download('twitter_samples')

# ---------------------- Data Preparation & Model Training ----------------------
# Load tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Cleaning function
def clean_tweet(tweet):
    tweet = re.sub(r'@\w+|#\w+|https\s+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

all_positive_tweets = [clean_tweet(tweet) for tweet in all_positive_tweets]
all_negative_tweets = [clean_tweet(tweet) for tweet in all_negative_tweets]

total_tweet = all_positive_tweets + all_negative_tweets
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
token = [tokenizer.tokenize(tweet) for tweet in total_tweet]

# Stemming & stopwords removal
stemmer = PorterStemmer()
data = []
tweets = []
for lst in token:
    tweet = []
    for word in lst:
        if word not in stopwords.words('english'):
            stem_word = stemmer.stem(word)
            data.append(stem_word)
            tweet.append(stem_word)
    tweets.append(tweet)

tweets_joined = [" ".join(tweet) for tweet in tweets]

# TF-IDF Vectorizer
vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=4000)
data_transformed = vec.fit_transform(tweets_joined)
x = data_transformed.toarray()
y = np.append(np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets)))

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x, y)

# ---------------------- Prediction Function ----------------------
def predict_sentiment(text):
    text = clean_tweet(text)
    tokens = tokenizer.tokenize(text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]
    processed_text = " ".join(processed_tokens)
    vector = vec.transform([processed_text]).toarray()
    prediction = model.predict(vector)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

# ---------------------- Streamlit Frontend ----------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="📝", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4B0082;'>📝 Tweet Sentiment Analyzer</h1>
        <p style='color: #6A5ACD;'>Type your tweet below and find out its sentiment!</p>
    </div>
""", unsafe_allow_html=True)

# Input box
user_input = st.text_area("Enter your tweet here:", height=120)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        result = predict_sentiment(user_input)
        if "Positive" in result:
            st.success(result)
        else:
            st.error(result)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <p style='color: #888888; font-size: 14px;'>Developed with ❤️ by Love Kumar</p>
    </div>
""", unsafe_allow_html=True)
