"""
Module 3: Feature Selection
Module 4: Model Building (Random Forest, KNN, Logistic Regression)
Module 5: Model Evaluation
------------------------------------------------------------------
Trains and compares 3 algorithms per disease using a 70/30 train/test
split, then saves the best model + scaler for each disease to /models.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from preprocess import load_diabetes, load_heart, scale_features

RANDOM_STATE = 42


def feature_importance_report(X, y, feature_names):
    """Module 3: rank features using a quick Random Forest importance scan."""
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def train_disease_models(X, y, disease_name):
    print(f"\n{'='*60}\n{disease_name.upper()}\n{'='*60}")

    # Module 3: Feature selection / importance
    importances = feature_importance_report(X, y, X.columns)
    print("Feature importance ranking:")
    print(importances.round(4).to_string())

    # 70% train / 30% test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} ({len(X_train)/len(X):.0%})  |  "
          f"Test size: {len(X_test)} ({len(X_test)/len(X):.0%})")

    # Scale (KNN and LogisticRegression need this; RF doesn't but it doesn't hurt)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Module 4: Train multiple algorithms
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics
        fitted[name] = model
        print(f"\n{name}: acc={metrics['accuracy']}  precision={metrics['precision']}  "
              f"recall={metrics['recall']}  f1={metrics['f1_score']}")
        print(f"  confusion_matrix={metrics['confusion_matrix']}")

    # Module 5: Pick best model by F1 score (balances precision/recall — important for medical data)
    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = fitted[best_name]
    print(f"\n>>> Best model for {disease_name}: {best_name} "
          f"(F1={results[best_name]['f1_score']})")

    # Save best model + scaler + feature order + all results
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{disease_name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{disease_name}_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, f"{disease_name}_meta.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "feature_order": list(X.columns),
            "results": results,
            "feature_importance": importances.round(4).to_dict(),
        }, f, indent=2)

    return best_name, results


if __name__ == "__main__":
    Xd, yd, _ = load_diabetes()
    train_disease_models(Xd, yd, "diabetes")

    Xh, yh, _ = load_heart()
    train_disease_models(Xh, yh, "heart")
