import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle

# Load your dataset
df = pd.read_csv("stress.csv", encoding="latin1")

# Replace with actual column names in your CSV
X = df['text']
y = df['label']

# Convert text to vectors
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = BernoulliNB()
model.fit(X_vec, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
