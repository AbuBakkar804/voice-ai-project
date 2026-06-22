"""
visualization.py
----------------
Plotting helpers for the Streamlit GUI and for training scripts.

Functions return Matplotlib Figure objects so Streamlit can render them
with st.pyplot(fig), and training scripts can save them with fig.savefig(...).

Includes:
  - Waveform
  - Spectrogram
  - MFCC heatmap
  - Mel spectrogram
  - Confusion matrix
  - Training accuracy / loss curves
  - Confidence bar chart

Author: Voice AI System
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for servers/Streamlit
import matplotlib.pyplot as plt
import itertools

SAMPLE_RATE = 22050


def plot_waveform(signal, sr=SAMPLE_RATE):
    """Plot amplitude over time — the basic 'shape' of the sound."""
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax, color="#2563eb")
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_spectrogram(signal, sr=SAMPLE_RATE):
    """Plot a log-frequency spectrogram (energy across frequency over time)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    stft = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    img = librosa.display.specshow(
        stft, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma"
    )
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_mfcc(signal, sr=SAMPLE_RATE, n_mfcc=40):
    """Plot the MFCC heatmap — the model's main 'view' of the voice timbre."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="viridis")
    ax.set_title("MFCC Features")
    ax.set_ylabel("MFCC Coefficients")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(signal, sr=SAMPLE_RATE):
    """Plot the Mel spectrogram (perceptually-scaled frequency)."""
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="inferno"
    )
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_confidence_bars(scores_dict, title="Confidence Scores"):
    """
    Horizontal bar chart of prediction confidences.

    Parameters
    ----------
    scores_dict : dict[str, float]  e.g. {'Happy': 0.7, 'Sad': 0.1, ...}
    """
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())

    fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(labels))))
    bars = ax.barh(labels, values, color="#2563eb")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Confidence")

    # Write the percentage next to each bar.
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                f"{value*100:.1f}%", va="center")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Render a confusion matrix as a labelled heatmap.

    cm : 2D np.ndarray from sklearn.metrics.confusion_matrix
    class_names : list of class label strings
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Write the count inside each cell; pick text colour for readability.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "d"), ha="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


def plot_training_history(history, title="Training History"):
    """
    Plot accuracy and loss curves from a Keras History object (or a dict).

    Works whether you pass `model.fit(...)` history or history.history dict.
    """
    hist = history.history if hasattr(history, "history") else history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Accuracy curve
    if "accuracy" in hist:
        ax1.plot(hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        ax1.plot(hist["val_accuracy"], label="validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss curve
    if "loss" in hist:
        ax2.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        ax2.plot(hist["val_loss"], label="validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig
