import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = joblib.load("logistic_regression_diabetes_model.joblib")

# Load scaler
scaler = joblib.load("scaler.joblib")

# Load feature names
feature_columns = joblib.load("feature_columns.joblib")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        input_df = pd.DataFrame([data])

        # Add missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Correct order
        input_df = input_df[feature_columns]

        # Scale
        scaled = scaler.transform(input_df)

        # Prediction
        pred = model.predict(scaled)[0]

        proba = model.predict_proba(scaled)[0]

        return jsonify({
            "prediction": int(pred),
            "probability_no_diabetes": float(proba[0]),
            "probability_diabetes": float(proba[1])
        })

    except Exception as e:

        return jsonify({"error": str(e)}),400


if __name__=="__main__":
    app.run(debug=True)
