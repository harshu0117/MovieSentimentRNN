import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
import os
MODEL_PATH = os.path.abspath("simplernn_imdb.h5")  # Get full path
model = load_model(MODEL_PATH)

# model = load_model('simplernn_imdb.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    if not words:
        return np.zeros((1, 500))  # Handle empty input safely

    encoded_review = [word_index.get(word, 2) for word in words]  # Keep index within range
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return np.array(padded_review, dtype=np.float32)  # Ensure correct type

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.4 else 'Negative'

        # Display result
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
    else:
        st.warning('⚠️ Please enter a valid movie review.')
