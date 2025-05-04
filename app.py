import streamlit as st
import joblib
import os

# Title
st.set_page_config(page_title="Stress Detection", page_icon="🧠")
st.title("🧠 Stress Detection App")

# Paths to model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load model and vectorizer safely
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        st.error("❌ Required files not found. Please upload 'model.pkl' and 'vectorizer.pkl'.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model/vectorizer: {e}")
    st.stop()

# Text input
user_input = st.text_area("Enter a sentence to analyze stress level:")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

            if prediction == 1:
                st.error("Prediction: Stressed 😥")
            else:
                st.success("Prediction: Not Stressed 😌")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
