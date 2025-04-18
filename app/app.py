from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and feature names
with open("model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

@app.route("/")
def home():
    return "Regression Model is Running"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "'features' key is missing"}), 400

    input_features = data["features"]

    if not isinstance(input_features, list) or not all(isinstance(item, dict) for item in input_features):
        return jsonify({"error": "'features' must be a list of dictionaries"}), 400

    df = pd.DataFrame(input_features)

    # Ensure the same columns (add missing with 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
