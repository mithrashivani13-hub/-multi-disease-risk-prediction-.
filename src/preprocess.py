"""
Module 2: Data Preprocessing
------------------------------------
Cleans raw CSVs, handles missing/invalid values, and prepares
clean feature/target arrays for both the Diabetes and Heart Disease datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DIABETES_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]


def load_diabetes(raw_path=None):
    if raw_path is None:
        raw_path = os.path.join(DATA_DIR, "diabetes_raw.csv")
    df = pd.read_csv(raw_path, header=None, names=DIABETES_COLS)

    # In this dataset, 0 is used as a placeholder for missing data in these
    # physiologically-impossible-to-be-zero columns. Replace with NaN, then
    # impute with the median (robust to outliers).
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    for col in zero_as_missing:
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y, df


def load_heart(raw_path=None):
    if raw_path is None:
        raw_path = os.path.join(DATA_DIR, "heart_raw.csv")
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df = df.drop_duplicates()
    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, df


def scale_features(X_train, X_test):
    """Fit StandardScaler on train only, transform both (avoids data leakage)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    Xd, yd, dfd = load_diabetes()
    print("Diabetes dataset:", dfd.shape, "| positive rate:", round(yd.mean(), 3))

    Xh, yh, dfh = load_heart()
    print("Heart dataset:", dfh.shape, "| positive rate:", round(yh.mean(), 3))
