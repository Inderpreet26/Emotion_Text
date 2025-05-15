# 🎭 Text Emotion Detection using XGBoost

This project uses a Machine Learning model trained with XGBoost to detect emotions from text input. It predicts emotions like **joy**, **sadness**, **anger**, **fear**, **love**, and **surprise** based on what the user writes.

## 🔍 Features
- Trained on 400K+ emotion-labeled sentences.
- Text preprocessing using TF-IDF.
- XGBoost for multi-class emotion classification.
- Interactive web app built using Streamlit.

## 🚀 Demo

Run the app locally:

```bash
# Activate virtual environment
.\emotion_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
