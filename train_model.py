import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
from model import SimpleFakeNewsDetector

print("🚀 Starting training...")

csv_file = "true.csv"

# Check current folder
print("Current folder:", os.getcwd())

# Check CSV exists
if not os.path.exists(csv_file):
    print(f"❌ CSV file '{csv_file}' not found!")
    exit()

# Load CSV
data = pd.read_csv(csv_file)
print("✅ Dataset loaded!")
print("Columns:", data.columns.tolist())
print("First 5 rows:\n", data.head())

# Check for required columns
if 'text' not in data.columns:
    print("❌ CSV must have 'text' column.")
    exit()

# Since we only have true news, we'll create a similarity-based model
# that compares new text against known true news patterns
X = data['text'].tolist()

# Vectorize all true news articles
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Create a simple model that stores the vectorized true news
# For prediction, we'll compare similarity to these patterns

# Create and save the model
model = SimpleFakeNewsDetector(vectorizer, X_tfidf)

print(f"✅ Model trained with {len(X)} true news articles")
print("📊 Model uses similarity-based detection")

# Save model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("💾 Model saved as fake_news_model.pkl")
