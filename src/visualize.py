"""
Module 9: Confusion Matrix & ROC Curve Visualization
------------------------------------------------------
Generates confusion matrix heatmaps (from saved metrics) and ROC curves
(re-run on the same held-out test split used in train.py, random_state=42,
so results reflect genuine unseen-data performance) for the best model of
each disease. Saves PNGs to /models for use in the report/poster.

Run with: python3 visualize.py
"""

import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from preprocess import load_diabetes, load_heart, scale_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
RANDOM_STATE = 42

LABELS = {
    "diabetes": ["No Diabetes", "Diabetes"],
    "heart":    ["No Heart Disease", "Heart Disease"],
}

LOADERS = {
    "diabetes": load_diabetes,
    "heart": load_heart,
}


def plot_confusion_matrix(disease, cm, model_name, out_path):
    cm = np.array(cm)
    labels = LABELS[disease]
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(
        f"{disease.capitalize()} — Confusion Matrix ({model_name})",
        fontsize=13, fontweight="bold", pad=14
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total
            color = "white" if count > thresh else "black"
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=13, fontweight="bold"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_roc_curve(disease, model_name, out_path):
    """Reproduces the exact test split from train.py, scores it with the
    already-trained saved model, and plots a real ROC curve (not synthetic)."""
    X, y, _ = LOADERS[disease]()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    # Re-fit a scaler on train only (matches train.py's leakage-safe approach)
    _, X_test_scaled, _ = scale_features(X_train, X_test)

    model = joblib.load(f"{MODEL_DIR}/{disease}_model.pkl")
    y_scores = model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#1B3A4B", linewidth=2.5,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#C5D2D9", linewidth=1.5, linestyle="--",
            label="Random guess (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(
        f"{disease.capitalize()} — ROC Curve ({model_name})",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  (AUC = {roc_auc:.4f})")
    return roc_auc


if __name__ == "__main__":
    auc_scores = {}
    for disease in ["diabetes", "heart"]:
        with open(f"{MODEL_DIR}/{disease}_meta.json") as f:
            meta = json.load(f)
        best = meta["best_model"]
        cm = meta["results"][best]["confusion_matrix"]

        plot_confusion_matrix(disease, cm, best, f"{MODEL_DIR}/{disease}_confusion_matrix.png")
        auc_scores[disease] = plot_roc_curve(disease, best, f"{MODEL_DIR}/{disease}_roc_curve.png")

    print("\nAUC summary:", json.dumps(auc_scores, indent=2))

