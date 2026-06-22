"""
model_builder.py
----------------
Defines the neural network architectures used across all training scripts,
plus GPU detection. Keeping the model definitions in one place means
train_gender.py, train_age.py and train_emotion.py stay short and consistent.

We use TensorFlow / Keras because it is the most beginner-friendly deep
learning framework while still being production-grade.

Author: Voice AI System
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def configure_gpu():
    """
    Detect and configure GPU(s) if available.

    'Memory growth' tells TensorFlow to allocate GPU memory gradually
    instead of grabbing all of it up front — friendlier on shared machines.

    Returns
    -------
    str : a human-readable device summary.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"GPU enabled: {len(gpus)} device(s) found."
        except RuntimeError as e:
            return f"GPU found but config failed: {e}"
    return "No GPU found — training on CPU."


def build_dense_classifier(input_dim, num_classes, name="dense_classifier"):
    """
    A fully-connected (dense) network for the 1D feature VECTOR.

    Good for: gender, age-group, and emotion when using mean-pooled features.
    Architecture: Input -> Dense(256) -> Dense(128) -> Dense(64) -> Softmax.
    Dropout and BatchNorm fight overfitting.

    Parameters
    ----------
    input_dim   : length of the feature vector
    num_classes : number of output classes
    """
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(input_dim,)))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_classifier(input_shape, num_classes, name="cnn_classifier"):
    """
    A 2D Convolutional Neural Network for the MEL SPECTROGRAM "image".

    CNNs are excellent at picking up local time-frequency patterns, which
    is why they tend to beat dense nets on emotion recognition.

    input_shape : (n_mels, max_frames, 1)
    """
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
