import streamlit as st
import joblib
import xgboost
import numpy as np

# Load vectorizer and model
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/xgb_emotion_model.pkl")

# Emotion labels 
emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# App UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🎭 Text Emotion Detection")
st.markdown("Enter a sentence to detect its **emotion** using a trained ML model.")

# User input
user_input = st.text_area("Enter text here", height=150)

# Predict
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)
        predicted_emotion = emotion_labels[int(prediction[0])]

        st.success(f"**Predicted Emotion:** {predicted_emotion.upper()} 🎉")
