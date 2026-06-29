import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model
model = joblib.load("logistic_regression_diabetes_model.joblib")

# Load scaler
scalar = joblib.load("scalar.joblib")

# Load feature names
feature_columns = joblib.load("feature_columns.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running successfully!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Add missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Arrange columns in training order
        input_df = input_df[feature_columns]

        # Scale input
        scaled_data = scalar.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_data)[0]

        # Prediction probability
        probability = model.predict_proba(scaled_data)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability_no_diabetes": round(float(probability[0]), 4),
            "probability_diabetes": round(float(probability[1]), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
