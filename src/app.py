"""
Module 7: Backend API
-----------------------
Flask REST API that the frontend (Figma-designed UI) calls to get
real-time multi-disease risk predictions.

Run with: python3 app.py
Then POST to http://localhost:5000/predict/diabetes
           or http://localhost:5000/predict/heart
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_diabetes, predict_heart

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately, e.g. from Figma/static host) to call this API


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/diabetes", methods=["POST"])
def diabetes_endpoint():
    data = request.get_json(force=True)
    required = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = predict_diabetes(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/heart", methods=["POST"])
def heart_endpoint():
    data = request.get_json(force=True)
    required = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = predict_heart(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
