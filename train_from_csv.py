"""
train_from_csv.py
-----------------
Trains gender + emotion sklearn MLPClassifier models directly from the
pre-extracted MFCC features in emotions.csv.

Saves .pkl artefacts that predict.py can load immediately:
  models/gender_model.pkl   gender_encoder.pkl   gender_scaler.pkl
  models/emotion_model.pkl  emotion_encoder.pkl  emotion_scaler.pkl

Usage:
    python train_from_csv.py
    python train_from_csv.py --csv emotions.csv --out models
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = "models"
CSV_FILE  = "emotions.csv"

# Map emotion names that differ from the app's display labels.
EMOTION_REMAP = {"calm": "neutral"}


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "labels"]
    X = df[feature_cols].values.astype(np.float32)
    labels = df["labels"].str.strip()
    return X, labels


def split_label(label):
    """'male_calm' → ('male', 'neutral').  Remaps calm→neutral."""
    parts = label.split("_", 1)
    gender  = parts[0]
    emotion = parts[1] if len(parts) > 1 else "neutral"
    emotion = EMOTION_REMAP.get(emotion, emotion)
    return gender, emotion


def train_classifier(X_train, X_test, y_train, y_test, class_names, name):
    print(f"\n--- Training {name} ({len(class_names)} classes: {list(class_names)}) ---")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train from pre-extracted CSV features")
    parser.add_argument("--csv", default=CSV_FILE, help="Path to emotions.csv")
    parser.add_argument("--out", default=MODEL_DIR, help="Output directory for .pkl files")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading features from '{args.csv}' …")
    X, raw_labels = load_csv(args.csv)
    print(f"  {len(X)} samples, {X.shape[1]} features each")

    gender_labels, emotion_labels = [], []
    for lbl in raw_labels:
        g, e = split_label(lbl)
        gender_labels.append(g)
        emotion_labels.append(e)

    print(f"  Gender classes : {sorted(set(gender_labels))}")
    print(f"  Emotion classes: {sorted(set(emotion_labels))}")

    # ── Gender ────────────────────────────────────────────────────────────
    g_enc    = LabelEncoder()
    g_y      = g_enc.fit_transform(gender_labels)
    g_scaler = StandardScaler()

    gX_tr, gX_te, gy_tr, gy_te = train_test_split(
        X, g_y, test_size=0.2, random_state=42, stratify=g_y
    )
    gX_tr_s = g_scaler.fit_transform(gX_tr)
    gX_te_s = g_scaler.transform(gX_te)

    g_model = train_classifier(gX_tr_s, gX_te_s, gy_tr, gy_te, g_enc.classes_, "gender")

    joblib.dump(g_model,  os.path.join(args.out, "gender_model.pkl"))
    joblib.dump(g_enc,    os.path.join(args.out, "gender_encoder.pkl"))
    joblib.dump(g_scaler, os.path.join(args.out, "gender_scaler.pkl"))
    print(f"  Saved: {args.out}/gender_model.pkl")

    # ── Emotion ───────────────────────────────────────────────────────────
    e_enc    = LabelEncoder()
    e_y      = e_enc.fit_transform(emotion_labels)
    e_scaler = StandardScaler()

    eX_tr, eX_te, ey_tr, ey_te = train_test_split(
        X, e_y, test_size=0.2, random_state=42, stratify=e_y
    )
    eX_tr_s = e_scaler.fit_transform(eX_tr)
    eX_te_s = e_scaler.transform(eX_te)

    e_model = train_classifier(eX_tr_s, eX_te_s, ey_tr, ey_te, e_enc.classes_, "emotion")

    joblib.dump(e_model,  os.path.join(args.out, "emotion_model.pkl"))
    joblib.dump(e_enc,    os.path.join(args.out, "emotion_encoder.pkl"))
    joblib.dump(e_scaler, os.path.join(args.out, "emotion_scaler.pkl"))
    print(f"  Saved: {args.out}/emotion_model.pkl")

    print("\nDone. Restart the Streamlit app -- AI models are now active.")


if __name__ == "__main__":
    main()
