import streamlit as st
import pickle
import os

# Check if model and vectorizer files exist
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    # Load model and vectorizer
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
else:
    st.error("Error: Model or Vectorizer file not found. Please upload the required files.")
    st.stop()  # Stops the execution of the app if files are not found

# Title
st.title("🧠 Stress Detection App")

# Input
user_input = st.text_area("Enter a sentence to analyze stress level:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("Prediction: Stressed 😥")
        else:
            st.success("Prediction: Not Stressed 😌")
