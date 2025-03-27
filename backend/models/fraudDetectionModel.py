import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../data/Credit_Card_Applications.csv")  # Make sure you have this CSV file

# Assume the last column is the target (fraud / not fraud)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target (0 = Legit, 1 = Fraud)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model to a file
joblib.dump(model, "fraud_model.pkl")

print("Model saved as fraud_model.pkl")
