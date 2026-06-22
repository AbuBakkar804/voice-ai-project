"""
train_all.py
------------
Single-command training script.  Trains the gender AND emotion models from
datasets/emotions.csv using scikit-learn MLPClassifier (no TensorFlow needed).
All artefacts are saved to models/ as .pkl files.

Usage:
    python train_all.py              # default settings
    python train_all.py --epochs 300

Dataset: datasets/emotions.csv
  - Columns 0-19  : 20 MFCC mean features
  - Column 'labels': "{gender}_{emotion}"  e.g. "male_angry", "female_calm"

Models produced:
    models/gender_model.pkl   + gender_encoder.pkl  + gender_scaler.pkl
    models/emotion_model.pkl  + emotion_encoder.pkl + emotion_scaler.pkl
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

EMOTIONS_CSV = "datasets/emotions.csv"
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(EMOTIONS_CSV)
    feature_cols = [c for c in df.columns if c != "labels"]
    X = df[feature_cols].values.astype(np.float32)
    df["gender"]  = df["labels"].str.split("_").str[0]
    df["emotion"] = df["labels"].str.split("_", n=1).str[1]
    return X, df["gender"].values, df["emotion"].values


def train_and_save(X, y_raw, tag, epochs):
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,          # L2 regularisation
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=True,
    )
    print(f"\n  Training {tag} model on {len(X_tr)} samples …")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    print(f"\n  {tag} validation accuracy: {acc*100:.1f}%")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    joblib.dump(clf, f"{MODEL_DIR}/{tag}_model.pkl")
    joblib.dump(le,  f"{MODEL_DIR}/{tag}_encoder.pkl")
    joblib.dump(sc,  f"{MODEL_DIR}/{tag}_scaler.pkl")
    print(f"  Saved: {MODEL_DIR}/{tag}_model.pkl\n")


def run(epochs=300):
    print(f"Loading {EMOTIONS_CSV} …")
    X, genders, emotions = load_data()
    print(f"  Samples={len(X)}  Features={X.shape[1]}")
    print(f"  Genders : {np.unique(genders)}")
    print(f"  Emotions: {np.unique(emotions)}\n")

    print("=" * 55)
    print("  GENDER MODEL")
    print("=" * 55)
    train_and_save(X, genders, "gender", epochs)

    print("=" * 55)
    print("  EMOTION MODEL")
    print("=" * 55)
    train_and_save(X, emotions, "emotion", epochs)

    print("=" * 55)
    print("  Done. All models saved to models/")
    print("  Age model: pitch heuristic (not enough labeled data to train).")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    run(epochs=args.epochs)
