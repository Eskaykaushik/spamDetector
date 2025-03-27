import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/combined_data.csv")  # Load the CSV file
df.columns = ["label", "text"]  # Rename columns

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Build pipeline (TF-IDF Vectorizer + Naïve Bayes Classifier)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Save trained model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as spam_model.pkl")
