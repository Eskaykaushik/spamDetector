from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

# Load trained model
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "")

    # Make prediction
    prediction = model.predict([email_text])[0]
    confidence = model.predict_proba([email_text])[0][prediction]

    return jsonify({"spam": bool(prediction), "confidence": round(confidence, 2)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

