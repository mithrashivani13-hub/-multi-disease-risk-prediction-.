"""
Module 6: Integration
----------------------
Loads the trained diabetes + heart models and exposes a single function
that takes a raw patient input dict and returns combined risk predictions.
"""

import json
import joblib
import numpy as np
import pandas as pd

import os
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

_cache = {}


def _load(disease):
    if disease not in _cache:
        model = joblib.load(f"{MODEL_DIR}/{disease}_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/{disease}_scaler.pkl")
        with open(f"{MODEL_DIR}/{disease}_meta.json") as f:
            meta = json.load(f)
        _cache[disease] = (model, scaler, meta)
    return _cache[disease]


def predict_diabetes(patient: dict):
    model, scaler, meta = _load("diabetes")
    order = meta["feature_order"]
    x = pd.DataFrame([[patient[f] for f in order]], columns=order)
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = float(model.predict_proba(x_scaled)[0][1])
    return {
        "disease": "Diabetes",
        "prediction": "At Risk" if pred == 1 else "Low Risk",
        "risk_probability": round(proba, 4),
        "model_used": meta["best_model"],
    }


def predict_heart(patient: dict):
    model, scaler, meta = _load("heart")
    order = meta["feature_order"]
    x = pd.DataFrame([[patient[f] for f in order]], columns=order)
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = float(model.predict_proba(x_scaled)[0][1])
    return {
        "disease": "Heart Disease",
        "prediction": "At Risk" if pred == 1 else "Low Risk",
        "risk_probability": round(proba, 4),
        "model_used": meta["best_model"],
    }


def predict_all(diabetes_input: dict, heart_input: dict):
    """Combined multi-disease prediction — the core integration step."""
    results = []
    if diabetes_input:
        results.append(predict_diabetes(diabetes_input))
    if heart_input:
        results.append(predict_heart(heart_input))
    return {"results": results}


if __name__ == "__main__":
    # Sample patient
    sample_diabetes = {
        "Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
        "SkinThickness": 25, "Insulin": 90, "BMI": 31.5,
        "DiabetesPedigreeFunction": 0.5, "Age": 45
    }
    sample_heart = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
    }
    print(json.dumps(predict_all(sample_diabetes, sample_heart), indent=2))
