import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model_and_vectorizer = joblib.load('model_and_vectorizer.joblib')
model = model_and_vectorizer['model']
vectorizer = model_and_vectorizer['vectorizer']

# Streamlit UI
st.title('Text Classification App')

# User input
user_input = st.text_area('Enter text for classification:', '')

if user_input:
    # Vectorize the user input
    user_input_vectorized = vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict(user_input_vectorized)

    # Display the result
    st.write('Prediction:', prediction[0])
