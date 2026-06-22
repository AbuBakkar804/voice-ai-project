"""
train_gender.py
---------------
Trains the GENDER classifier (Male vs Female).

Pipeline:
  1. Walk a dataset folder of audio files.
  2. Read the label from the folder name (datasets/gender/male, /female).
  3. Extract a feature vector from each file.
  4. Train a dense neural network.
  5. Evaluate: accuracy, classification report, confusion matrix.
  6. Save the model + label encoder + scaler + plots.

USAGE
-----
    python train_gender.py --data_dir datasets/gender --epochs 50

Expected dataset layout:
    datasets/gender/
        male/    *.wav
        female/  *.wav

If you do not yet have data, run with --demo to train on synthetic data so
you can see the whole pipeline work end-to-end.

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

# Where trained artefacts are written.
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")


def load_dataset(data_dir):
    """
    Load all audio files under data_dir, using subfolder names as labels.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : list[str]   labels
    """
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


def make_demo_data(n_per_class=120, n_features=102):
    """
    Generate synthetic data so the pipeline runs without real audio.

    We give the two classes slightly different means so the model can
    actually learn something (useful for testing the full flow).
    """
    rng = np.random.default_rng(42)
    male = rng.normal(loc=-0.5, scale=1.0, size=(n_per_class, n_features))
    female = rng.normal(loc=0.5, scale=1.0, size=(n_per_class, n_features))
    X = np.vstack([male, female]).astype(np.float32)
    y = ["male"] * n_per_class + ["female"] * n_per_class
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train gender classifier")
    parser.add_argument("--data_dir", default="datasets/gender")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--demo", action="store_true",
                        help="Train on synthetic data instead of real files")
    args = parser.parse_args()

    print(configure_gpu())

    # 1) Load data ---------------------------------------------------------
    if args.demo or not os.path.isdir(args.data_dir):
        print("Using DEMO synthetic data.")
        X, y = make_demo_data()
    else:
        X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        raise SystemExit("No data loaded. Add audio files or use --demo.")

    # 2) Encode labels & scale features -----------------------------------
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 3) Build & train -----------------------------------------------------
    model = build_dense_classifier(
        input_dim=X.shape[1],
        num_classes=len(encoder.classes_),
        name="gender_model",
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "gender_model.h5"),
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

    # 4) Evaluate ----------------------------------------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plot_confusion_matrix(cm, encoder.classes_, "Gender Confusion Matrix")
    fig_cm.savefig(os.path.join(MODEL_DIR, "gender_confusion_matrix.png"), dpi=120)

    fig_hist = plot_training_history(history, "Gender Training History")
    fig_hist.savefig(os.path.join(MODEL_DIR, "gender_training_history.png"), dpi=120)

    # 5) Save artefacts ----------------------------------------------------
    model.save(os.path.join(MODEL_DIR, "gender_model.h5"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "gender_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "gender_scaler.pkl"))
    print(f"\nSaved gender model + encoder + scaler to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
