"""
preprocessing.py
----------------
Audio cleaning and preparation utilities used before feature extraction.

Includes:
  - Silence trimming
  - Background noise reduction
  - Voice Activity Detection (VAD)
  - Speaking-rate estimation
  - Pitch-band classification

Good preprocessing makes the models far more accurate because it removes
junk (silence, hiss) that would otherwise confuse them.

Author: Voice AI System
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050


def trim_silence(signal, top_db=25):
    """
    Remove leading/trailing silence.

    top_db = anything quieter than (max - top_db) decibels is treated as
    silence. Lower top_db = more aggressive trimming.
    """
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def reduce_noise(signal, sr=SAMPLE_RATE):
    """
    Simple spectral-gating noise reduction.

    Idea: estimate the noise profile from the quietest parts of the signal,
    then subtract that profile from every frame. This is a lightweight,
    dependency-free version of what libraries like `noisereduce` do.
    """
    # Short-time Fourier transform -> complex spectrogram.
    stft = librosa.stft(signal)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Estimate the noise floor as the 10th percentile of each frequency bin.
    noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)

    # Subtract the noise floor; clip negatives to zero (can't have neg energy).
    cleaned_mag = np.maximum(magnitude - noise_profile, 0.0)

    # Rebuild the complex spectrogram and invert back to a waveform.
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    cleaned_signal = librosa.istft(cleaned_stft)
    return cleaned_signal


def voice_activity_detection(signal, sr=SAMPLE_RATE, frame_ms=25, threshold=0.02):
    """
    Detect which frames actually contain speech (vs silence/noise).

    Uses short-term RMS energy: frames louder than `threshold` are "voiced".

    Returns
    -------
    voiced_signal : np.ndarray
        Concatenation of only the voiced frames.
    speech_ratio : float
        Fraction of the clip that contained speech (0..1).
    """
    frame_length = int(sr * frame_ms / 1000)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Boolean mask of voiced frames.
    voiced_frames = rms > threshold
    speech_ratio = float(np.mean(voiced_frames)) if len(voiced_frames) else 0.0

    # Rebuild a signal containing only voiced samples.
    voiced_samples = []
    for i, is_voiced in enumerate(voiced_frames):
        if is_voiced:
            start = i * hop_length
            end = start + frame_length
            voiced_samples.extend(signal[start:end])

    voiced_signal = np.array(voiced_samples) if voiced_samples else signal
    return voiced_signal, speech_ratio


def estimate_speaking_rate(signal, sr=SAMPLE_RATE):
    """
    Estimate speaking rate by counting energy "peaks" (rough syllable count)
    and dividing by the voiced duration.

    Returns
    -------
    label : str   -> 'Slow' | 'Normal' | 'Fast'
    syllables_per_sec : float
    """
    # Onset envelope highlights moments where new sounds begin.
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3,
        post_avg=5, delta=0.3, wait=5,
    )

    duration = max(len(signal) / sr, 1e-6)
    syllables_per_sec = len(peaks) / duration

    # Typical conversational speech is ~3-5 syllables/sec.
    if syllables_per_sec < 2.5:
        label = "Slow"
    elif syllables_per_sec <= 5.0:
        label = "Normal"
    else:
        label = "Fast"

    return label, float(syllables_per_sec)


def classify_pitch_band(pitch_hz):
    """
    Turn a raw pitch value (Hz) into a human-readable band.

    Returns
    -------
    str -> 'Low' | 'Medium' | 'High'
    """
    if pitch_hz <= 0:
        return "Unknown"
    if pitch_hz < 150:
        return "Low"
    elif pitch_hz < 220:
        return "Medium"
    else:
        return "High"


def preprocess_pipeline(signal, sr=SAMPLE_RATE, denoise=True):
    """
    Full cleaning pipeline applied before feature extraction.

    Steps: trim silence -> (optional) denoise -> VAD.

    Returns
    -------
    clean_signal : np.ndarray
    speech_ratio : float
    """
    signal = trim_silence(signal)
    if len(signal) < int(sr * 0.3):
        raise ValueError(
            "Recording is too short or silent. "
            "Please speak for at least half a second and try again."
        )
    if denoise:
        signal = reduce_noise(signal, sr)
    signal, speech_ratio = voice_activity_detection(signal, sr)
    if len(signal) == 0:
        raise ValueError(
            "No speech detected in the recording. "
            "Please speak clearly and try again."
        )
    return signal, speech_ratio
