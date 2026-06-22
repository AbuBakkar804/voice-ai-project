"""
predict.py — inference engine.
Gender + emotion use sklearn MLPClassifier models (20-D MFCC vector).
Age falls back to a pitch heuristic (insufficient labeled data).
"""

import os
import numpy as np

from utils.feature_extraction import (
    load_audio, extract_mfcc_vector, extract_feature_vector, extract_pitch,
)
from utils.preprocessing import (
    preprocess_pipeline, estimate_speaking_rate, classify_pitch_band,
)

MODEL_DIR = "models"

AGE_RANGE_HINTS = {
    "child": "approx 5-12 yrs",
    "teen":  "approx 13-19 yrs",
    "young": "approx 20-40 yrs",
    "old":   "approx 40+ yrs",
}


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


class VoiceAnalyzer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir

        self.gender_model   = _load(f"{model_dir}/gender_model.pkl")
        self.gender_encoder = _load(f"{model_dir}/gender_encoder.pkl")
        self.gender_scaler  = _load(f"{model_dir}/gender_scaler.pkl")

        self.age_model   = _load(f"{model_dir}/age_model.pkl")
        self.age_encoder = _load(f"{model_dir}/age_encoder.pkl")
        self.age_scaler  = _load(f"{model_dir}/age_scaler.pkl")

        self.emotion_model   = _load(f"{model_dir}/emotion_model.pkl")
        self.emotion_encoder = _load(f"{model_dir}/emotion_encoder.pkl")
        self.emotion_scaler  = _load(f"{model_dir}/emotion_scaler.pkl")

    def _sklearn_predict(self, model, encoder, scaler, vec):
        """Run an sklearn MLP and return (label, confidence, scores_dict)."""
        x     = scaler.transform(vec.reshape(1, -1))
        probs = model.predict_proba(x)[0]
        idx   = int(np.argmax(probs))
        label = encoder.inverse_transform([idx])[0]
        scores = {
            encoder.inverse_transform([i])[0].capitalize(): float(p)
            for i, p in enumerate(probs)
        }
        return label.capitalize(), float(probs[idx]), scores

    def _predict_gender(self, mfcc_vector, pitch):
        if self.gender_model and self.gender_encoder and self.gender_scaler:
            label, conf, scores = self._sklearn_predict(
                self.gender_model, self.gender_encoder, self.gender_scaler, mfcc_vector)
            return {"value": label, "confidence": conf, "scores": scores, "source": "model"}
        if pitch <= 0:
            return {"value": "Unknown", "confidence": 0.0, "scores": {}, "source": "heuristic"}
        is_female = pitch >= 165
        conf = float(min(0.5 + abs(pitch - 165) / 200, 0.95))
        return {
            "value":      "Female" if is_female else "Male",
            "confidence": conf,
            "scores":     {"Female": conf if is_female else 1-conf,
                           "Male":   1-conf if is_female else conf},
            "source": "heuristic",
        }

    def _predict_age(self, feature_vector, pitch, gender="unknown"):
        if self.age_model and self.age_encoder and self.age_scaler:
            label, conf, scores = self._sklearn_predict(
                self.age_model, self.age_encoder, self.age_scaler, feature_vector)
            return {"value": label, "range": AGE_RANGE_HINTS.get(label.lower(), ""),
                    "confidence": conf, "scores": scores, "source": "model"}

        # ── multi-feature heuristic ────────────────────────────────────────
        # feature_vector layout (102-D from extract_feature_vector):
        #   [0:40]  MFCC mean   [40:80] MFCC std
        #   [80:92] Chroma      [92:99] Spectral contrast
        #   [99]    ZCR mean    [100]   RMS mean    [101] Pitch
        n = len(feature_vector)
        zcr      = float(feature_vector[99])  if n > 99 else 0.06
        mfcc1    = float(feature_vector[1])   if n > 1  else -10.0
        # mean MFCC std across time → higher = more dynamic / expressive voice
        mfcc_dyn = float(np.mean(feature_vector[40:80])) if n > 80 else 10.0

        s = {"child": 0.0, "teen": 0.0, "young": 0.0, "old": 0.0}

        # ── 1. Pitch (gender-aware thresholds) ────────────────────────────
        # Male:   child>190, teen≈155, young≈135, old≈108
        # Female: child>270, teen≈245, young≈200, old≈168
        # Unknown: averaged bands
        if pitch > 0:
            if gender == "male":
                s["child"] += max(0.0, (pitch - 190) / 60)
                s["teen"]  += max(0.0, 1.0 - abs(pitch - 155) / 55)
                s["young"] += max(0.0, 1.0 - abs(pitch - 135) / 55)
                s["old"]   += max(0.0, 1.0 - abs(pitch - 108) / 45)
                zcr_lo, zcr_hi = 0.02, 0.09
            elif gender == "female":
                s["child"] += max(0.0, (pitch - 270) / 55)
                s["teen"]  += max(0.0, 1.0 - abs(pitch - 245) / 50)
                s["young"] += max(0.0, 1.0 - abs(pitch - 200) / 55)
                s["old"]   += max(0.0, 1.0 - abs(pitch - 168) / 45)
                zcr_lo, zcr_hi = 0.04, 0.14
            else:
                s["child"] += max(0.0, (pitch - 230) / 65)
                s["teen"]  += max(0.0, 1.0 - abs(pitch - 215) / 65)
                s["young"] += max(0.0, 1.0 - abs(pitch - 165) / 80)
                s["old"]   += max(0.0, 1.0 - abs(pitch - 125) / 60)
                zcr_lo, zcr_hi = 0.03, 0.12
        else:
            s["young"] += 1.5           # no pitch detected → assume adult
            zcr_lo, zcr_hi = 0.03, 0.12

        # ── 2. ZCR: low ZCR = breathier / older voice ─────────────────────
        z = max(0.0, min(1.0, (zcr - zcr_lo) / max(zcr_hi - zcr_lo, 1e-6)))
        s["child"] += z * 0.5
        s["young"] += (z + 0.5) * 0.4   # moderate ZCR = adult prime
        s["old"]   += (1.0 - z) * 1.2   # low ZCR = breathiness → older

        # ── 3. MFCC[1] spectral tilt: very negative = more bass = older ───
        # Typical range: -30 to +10
        tilt = max(0.0, min(1.0, (-mfcc1) / 20.0))   # 0=bright, 1=dark
        s["old"]   += tilt * 0.8
        s["child"] += (1.0 - tilt) * 0.6

        # ── 4. Voice dynamics: lower MFCC std mean = more monotone = older ─
        dyn = max(0.0, min(1.0, (mfcc_dyn - 3.0) / 20.0))
        s["young"] += dyn * 0.5
        s["old"]   += (1.0 - dyn) * 0.6

        # ── normalize to probabilities ─────────────────────────────────────
        total = sum(s.values()) + 1e-9
        probs = {k: v / total for k, v in s.items()}
        group = max(probs, key=probs.get)
        conf  = min(0.85, max(0.42, probs[group]))

        return {
            "value":      group.capitalize(),
            "range":      AGE_RANGE_HINTS.get(group, ""),
            "confidence": conf,
            "scores":     {k.capitalize(): round(v, 3) for k, v in probs.items()},
            "source":     "heuristic",
        }

    def _predict_emotion(self, mfcc_vector):
        if self.emotion_model and self.emotion_encoder and self.emotion_scaler:
            label, conf, scores = self._sklearn_predict(
                self.emotion_model, self.emotion_encoder, self.emotion_scaler, mfcc_vector)
            return {"value": label, "confidence": conf, "scores": scores, "source": "model"}
        return {"value": "Neutral", "confidence": 0.5, "scores": {"Neutral": 0.5}, "source": "heuristic"}

    def analyze(self, file_path, denoise=True):
        raw_signal, sr = load_audio(file_path)
        clean_signal, speech_ratio = preprocess_pipeline(raw_signal, sr, denoise)

        mfcc_vector    = extract_mfcc_vector(clean_signal, sr)
        feature_vector = extract_feature_vector(clean_signal, sr)
        pitch          = extract_pitch(clean_signal, sr)

        gender  = self._predict_gender(mfcc_vector, pitch)
        age     = self._predict_age(feature_vector, pitch, gender["value"].lower())
        emotion = self._predict_emotion(mfcc_vector)
        rate_label, syl_per_sec = estimate_speaking_rate(clean_signal, sr)

        return {
            "gender":        gender,
            "age":           age,
            "emotion":       emotion,
            "pitch":         {"hz": round(pitch, 1), "band": classify_pitch_band(pitch)},
            "speaking_rate": {"label": rate_label, "syllables_per_sec": round(syl_per_sec, 2)},
            "speech_ratio":  round(speech_ratio, 2),
            "raw_signal":    raw_signal,
            "sr":            sr,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>"); raise SystemExit(1)
    r = VoiceAnalyzer().analyze(sys.argv[1])
    for k in ("gender", "age", "emotion"):
        v = r[k]
        print(f"{k:10}: {v['value']} ({v['confidence']*100:.1f}%) [{v['source']}]")
    print(f"pitch     : {r['pitch']['hz']} Hz  ({r['pitch']['band']})")
    print(f"rate      : {r['speaking_rate']['label']}  ({r['speaking_rate']['syllables_per_sec']} syl/s)")
