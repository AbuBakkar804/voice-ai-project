"""
feature_extraction.py
----------------------
Extracts acoustic features from audio signals using Librosa.

These features are the numerical "fingerprint" of a voice that our ML models
learn from. Each feature captures a different aspect of the sound:

  - MFCC               : timbre / how the voice "sounds" (most important for speech)
  - Chroma             : pitch class energy (relates to tone/melody)
  - Spectral Contrast  : difference between peaks and valleys in the spectrum
  - Zero Crossing Rate : how noisy/voiced the signal is
  - RMS Energy         : loudness over time
  - Pitch (F0)         : fundamental frequency (key for gender/age)
  - Mel Spectrogram    : 2D time-frequency image (used by CNN models)

Author: Voice AI System
"""

import numpy as np
import librosa

# Standard sample rate we resample everything to. 22050 Hz is a common
# default in Librosa and is enough to capture human speech (which lives
# mostly below 8 kHz).
SAMPLE_RATE = 22050

# Number of MFCC coefficients to keep. 40 is a typical, robust choice.
N_MFCC = 40


def load_audio(file_path, sr=SAMPLE_RATE):
    """
    Load an audio file from disk and resample to a fixed sample rate.

    Parameters
    ----------
    file_path : str
        Path to a .wav or .mp3 file.
    sr : int
        Target sample rate.

    Returns
    -------
    signal : np.ndarray
        1D mono audio waveform.
    sr : int
        The sample rate actually used.
    """
    # mono=True mixes stereo down to one channel so every file is consistent.
    signal, sr = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr


def extract_pitch(signal, sr=SAMPLE_RATE):
    """
    Estimate the fundamental frequency (pitch / F0) of the voice.

    Pitch is one of the strongest cues for gender:
    typically men ~85-180 Hz, women ~165-255 Hz.

    Returns the mean pitch in Hz (ignoring unvoiced frames).
    """
    # pyin is a robust pitch tracker. fmin/fmax bound the human voice range.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal,
        fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
        fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
        sr=sr,
    )
    # f0 contains NaN where the frame is unvoiced (silence/consonants).
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) == 0:
        return 0.0
    return float(np.mean(f0_clean))


def extract_feature_vector(signal, sr=SAMPLE_RATE):
    """
    Build a single fixed-length feature vector for "classic" ML models
    (e.g. scikit-learn / dense neural nets).

    We compute each feature over time, then take the MEAN across time so
    that audio clips of different lengths all produce the same vector size.

    Returns
    -------
    np.ndarray of shape (N,) where N is the total number of features.
    """
    features = []

    # 1) MFCC --------------------------------------------------------------
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))          # mean of each coefficient
    features.extend(np.std(mfcc, axis=1))           # spread adds extra info

    # 2) Chroma ------------------------------------------------------------
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.extend(np.mean(chroma, axis=1))        # 12 values

    # 3) Spectral Contrast -------------------------------------------------
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    features.extend(np.mean(contrast, axis=1))      # 7 values

    # 4) Zero Crossing Rate ------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    features.append(float(np.mean(zcr)))

    # 5) RMS Energy --------------------------------------------------------
    rms = librosa.feature.rms(y=signal)
    features.append(float(np.mean(rms)))

    # 6) Pitch -------------------------------------------------------------
    features.append(extract_pitch(signal, sr))

    return np.array(features, dtype=np.float32)


def extract_mel_spectrogram(signal, sr=SAMPLE_RATE, n_mels=128, max_frames=128):
    """
    Build a 2D Mel spectrogram "image" for CNN-based models.

    The output is padded or truncated to a fixed (n_mels x max_frames) shape
    so it can be fed to a convolutional network in batches.

    Returns
    -------
    np.ndarray of shape (n_mels, max_frames)
    """
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    # Convert raw power to decibels (log scale) — closer to how humans hear.
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Make every spectrogram the same width (time axis).
    if mel_db.shape[1] < max_frames:
        pad_width = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :max_frames]

    return mel_db.astype(np.float32)


def extract_mfcc_vector(signal, sr=SAMPLE_RATE, n_mfcc=20):
    """
    Return the mean of each MFCC coefficient across time — a compact 20-D
    vector that matches the format used in datasets/emotions.csv.

    This is the feature representation used by the trained gender and emotion
    dense models.  Keep n_mfcc=20 in sync with how emotions.csv was built.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).astype(np.float32)


def get_all_features(file_path):
    """
    Convenience wrapper: load a file and return BOTH representations.

    Returns
    -------
    dict with keys:
        'vector'    -> 1D feature vector
        'mel'       -> 2D mel spectrogram
        'signal'    -> raw waveform (for plotting)
        'sr'        -> sample rate
        'pitch'     -> mean pitch in Hz
    """
    signal, sr = load_audio(file_path)
    return {
        "vector": extract_feature_vector(signal, sr),
        "mel": extract_mel_spectrogram(signal, sr),
        "signal": signal,
        "sr": sr,
        "pitch": extract_pitch(signal, sr),
    }
