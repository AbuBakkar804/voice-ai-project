"""
train_emotion.py
----------------
Trains the EMOTION recognition model.

Classes: Happy | Sad | Angry | Neutral | Fear | Surprise

This one uses a 2D CNN on Mel spectrograms because emotion lives in subtle
time-frequency patterns that convolutional layers capture better than a
mean-pooled vector.

It also supports RAVDESS-style filenames. RAVDESS encodes emotion in the
3rd field of the filename, e.g. "03-01-05-01-02-01-12.wav" -> emotion id 05.
If you point --data_dir at a RAVDESS folder and pass --ravdess, labels are
parsed automatically from filenames.

USAGE
-----
    # Folder-per-class layout:
    python train_emotion.py --data_dir datasets/emotion --epochs 80

    # RAVDESS filename layout:
    python train_emotion.py --data_dir datasets/RAVDESS --ravdess --epochs 80

    # No data yet:
    python train_emotion.py --demo

Author: Voice AI System
"""

import os
import argparse
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.feature_extraction import load_audio, extract_mel_spectrogram
from utils.model_builder import build_cnn_classifier, configure_gpu
from utils.visualization import plot_confusion_matrix, plot_training_history

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

# RAVDESS emotion-id -> our label. (01=neutral,02=calm,03=happy,04=sad,
# 05=angry,06=fearful,07=disgust,08=surprised). We map to our 6 classes
# and skip 'calm'/'disgust' which are outside our set.
RAVDESS_MAP = {
    "01": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "08": "surprise",
}


def load_folder_dataset(data_dir):
    """Folder-per-class layout: datasets/emotion/happy/*.wav etc."""
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
            try:
                signal, sr = load_audio(os.path.join(folder, fname))
                X.append(extract_mel_spectrogram(signal, sr))
                y.append(label)
            except Exception as e:
                print(f"    skipped {fname}: {e}")
    return X, y


def load_ravdess_dataset(data_dir):
    """RAVDESS layout: parse emotion id from each filename (recursively)."""
    X, y = [], []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith(AUDIO_EXTS):
                continue
            parts = fname.split("-")
            if len(parts) < 3:
                continue
            emotion_id = parts[2]
            label = RAVDESS_MAP.get(emotion_id)
            if label is None:
                continue  # skip calm/disgust
            try:
                signal, sr = load_audio(os.path.join(root, fname))
                X.append(extract_mel_spectrogram(signal, sr))
                y.append(label)
            except Exception as e:
                print(f"    skipped {fname}: {e}")
    print(f"Loaded {len(X)} RAVDESS clips.")
    return X, y


def make_demo_data(n_per_class=40, shape=(128, 128)):
    """Synthetic spectrogram-like data for 6 classes."""
    rng = np.random.default_rng(11)
    X, y = [], []
    for i, emo in enumerate(EMOTIONS):
        for _ in range(n_per_class):
            base = rng.normal(loc=i * 0.5, scale=1.0, size=shape)
            X.append(base.astype(np.float32))
            y.append(emo)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train emotion recogniser")
    parser.add_argument("--data_dir", default="datasets/emotion")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ravdess", action="store_true",
                        help="Parse labels from RAVDESS filenames")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    print(configure_gpu())

    # 1) Load --------------------------------------------------------------
    if args.demo or not os.path.isdir(args.data_dir):
        print("Using DEMO synthetic data.")
        X, y = make_demo_data()
    elif args.ravdess:
        X, y = load_ravdess_dataset(args.data_dir)
    else:
        X, y = load_folder_dataset(args.data_dir)

    if len(X) == 0:
        raise SystemExit("No data loaded. Add audio files or use --demo.")

    # 2) Shape data for CNN: (samples, n_mels, frames, 1) ------------------
    X = np.array(X)
    X = X[..., np.newaxis]  # add channel dimension
    # Normalise to roughly 0..1 for stable training.
    X = (X - X.min()) / (X.max() - X.min() + 1e-9)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 3) Build & train -----------------------------------------------------
    model = build_cnn_classifier(
        input_shape=X.shape[1:],
        num_classes=len(encoder.classes_),
        name="emotion_model",
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "emotion_model.h5"),
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
    fig_cm = plot_confusion_matrix(cm, encoder.classes_, "Emotion Confusion Matrix")
    fig_cm.savefig(os.path.join(MODEL_DIR, "emotion_confusion_matrix.png"), dpi=120)

    fig_hist = plot_training_history(history, "Emotion Training History")
    fig_hist.savefig(os.path.join(MODEL_DIR, "emotion_training_history.png"), dpi=120)

    # 5) Save --------------------------------------------------------------
    model.save(os.path.join(MODEL_DIR, "emotion_model.h5"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "emotion_encoder.pkl"))
    # Save the spectrogram input shape so predict.py can rebuild correctly.
    joblib.dump({"input_shape": X.shape[1:]},
                os.path.join(MODEL_DIR, "emotion_meta.pkl"))
    print(f"\nSaved emotion model + encoder to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
