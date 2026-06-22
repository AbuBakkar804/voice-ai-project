"""
train_age.py
------------
Trains the AGE-GROUP classifier.

As requested, age is treated as recognising a *person category* rather than
an exact number:

    Child | Teen | Young | Old

(These map loosely to roughly: Child <13, Teen 13-19, Young 20-40, Old 40+.)
The predict module can still surface an approximate age range derived from
the predicted group + pitch, but the model itself classifies the band.

Pipeline mirrors train_gender.py.

USAGE
-----
    python train_age.py --data_dir datasets/age --epochs 60
    python train_age.py --demo

Expected layout:
    datasets/age/
        child/  *.wav
        teen/   *.wav
        young/  *.wav
        old/    *.wav

Author: Voice AI System
"""

import os
import argparse
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.feature_extraction import load_audio, extract_feature_vector
from utils.model_builder import build_dense_classifier, configure_gpu
from utils.visualization import plot_confusion_matrix, plot_training_history

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

# The fixed set of age groups we support (order is just for readability).
AGE_GROUPS = ["child", "teen", "young", "old"]


def load_dataset(data_dir):
    """Load audio files; subfolder name = age group label."""
    X, y = [], []
    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    print(f"Found classes: {classes}")

    for label in classes:
        folder = os.path.join(data_dir, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTS)]
        print(f"  {label}: {len(files)} files")
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                signal, sr = load_audio(path)
                X.append(extract_feature_vector(signal, sr))
                y.append(label)
            except Exception as e:
                print(f"    skipped {fname}: {e}")
    return np.array(X), y


def make_demo_data(n_per_class=80, n_features=102):
    """Synthetic 4-class data so the pipeline runs without real audio."""
    rng = np.random.default_rng(7)
    blocks, labels = [], []
    for i, group in enumerate(AGE_GROUPS):
        # Each group gets its own mean so classes are separable.
        center = (i - 1.5) * 0.8
        blocks.append(rng.normal(loc=center, scale=1.0,
                                 size=(n_per_class, n_features)))
        labels += [group] * n_per_class
    X = np.vstack(blocks).astype(np.float32)
    return X, labels


def main():
    parser = argparse.ArgumentParser(description="Train age-group classifier")
    parser.add_argument("--data_dir", default="datasets/age")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    print(configure_gpu())

    if args.demo or not os.path.isdir(args.data_dir):
        print("Using DEMO synthetic data.")
        X, y = make_demo_data()
    else:
        X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        raise SystemExit("No data loaded. Add audio files or use --demo.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = build_dense_classifier(
        input_dim=X.shape[1],
        num_classes=len(encoder.classes_),
        name="age_model",
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "age_model.h5"),
                        save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plot_confusion_matrix(cm, encoder.classes_, "Age Confusion Matrix")
    fig_cm.savefig(os.path.join(MODEL_DIR, "age_confusion_matrix.png"), dpi=120)

    fig_hist = plot_training_history(history, "Age Training History")
    fig_hist.savefig(os.path.join(MODEL_DIR, "age_training_history.png"), dpi=120)

    model.save(os.path.join(MODEL_DIR, "age_model.h5"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "age_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "age_scaler.pkl"))
    print(f"\nSaved age model + encoder + scaler to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
