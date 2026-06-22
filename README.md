# 🎙️ AI Voice Analysis System

An end-to-end deep-learning system that analyzes a person's voice and predicts
**gender**, **age group**, and **emotion**, plus **pitch**, **speaking rate**,
voice-activity, visualizations, history logging, and PDF/CSV export — all wrapped
in a professional **Streamlit** GUI.

Built to be **beginner-friendly** yet **portfolio / final-year-project grade**.

---

## ✨ Features

| Category | What it does |
|---|---|
| **Gender** | Male / Female (deep model + pitch fallback) |
| **Age group** | Child / Teen / Young / Old (+ approx age range) |
| **Emotion** | Happy, Sad, Angry, Neutral, Fear, Surprise (CNN on mel spectrograms) |
| **Pitch** | Mean F0 in Hz + Low/Medium/High band |
| **Speaking rate** | Slow / Normal / Fast (syllables per second) |
| **Confidence** | Per-task confidence scores + bar charts |
| **Visualizations** | Waveform, Spectrogram, MFCC heatmap |
| **Preprocessing** | Silence trim, noise reduction, Voice Activity Detection |
| **History** | Auto-logged dashboard with distribution charts |
| **Export** | One-click PDF and CSV reports |
| **GPU** | Auto-detected and enabled if available |

> If a model isn't trained yet, the app transparently falls back to rule-based
> heuristics and labels each result's source, so the GUI always works for demos.

---

## 📂 Project structure

```
voice_ai_project/
├── app.py                  # Streamlit GUI (upload / record / results / history)
├── predict.py              # Inference engine (VoiceAnalyzer class)
├── train_gender.py         # Trains gender classifier
├── train_age.py            # Trains age-group classifier
├── train_emotion.py        # Trains emotion CNN (supports RAVDESS filenames)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── datasets/               # Your audio data (folder-per-class) + dataset guide
├── models/                 # Saved models, encoders, scalers, plots
├── history/                # history.csv (auto-created)
├── exports/                # optional manual exports
└── utils/
    ├── __init__.py
    ├── feature_extraction.py  # MFCC, chroma, contrast, ZCR, RMS, pitch, mel
    ├── preprocessing.py       # trim, denoise, VAD, speaking rate, pitch band
    ├── visualization.py       # all plots (GUI + training)
    ├── model_builder.py       # network architectures + GPU config
    └── export_utils.py        # history logging + PDF/CSV export
```

### What each file does

- **app.py** — the user interface. Three tabs: Analyze (upload or record audio,
  run analysis, view results + charts, download reports), History (dashboard),
  About (model status).
- **predict.py** — loads the trained models once and exposes
  `VoiceAnalyzer.analyze(file)`. Falls back to heuristics if a model is missing.
- **train_gender.py / train_age.py / train_emotion.py** — load a dataset,
  extract features, train, evaluate (accuracy, classification report,
  confusion matrix, training curves), and save all artefacts to `models/`.
- **utils/feature_extraction.py** — converts raw audio into the numeric features
  the models learn from.
- **utils/preprocessing.py** — cleans audio (trim/denoise/VAD) and computes
  speaking rate and pitch band.
- **utils/visualization.py** — every Matplotlib figure used in the app and during
  training.
- **utils/model_builder.py** — the dense and CNN architectures, plus GPU setup.
- **utils/export_utils.py** — writes history rows and builds CSV/PDF exports.

---

## 🚀 Quick start

### 1. Install

```bash
cd voice_ai_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app immediately (no training needed)

```bash
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501). Upload a WAV/MP3 or
record from your mic. Untrained tasks use heuristics until you train models.

---

## 🧠 Training

Put data into `datasets/` (see `datasets/README.md`), then:

### Step 1 — Download a dataset (free, from Kaggle)

| You want | Download from Kaggle | Search term |
|---|---|---|
| **Emotion** | RAVDESS (clean, ~1.5k clips) | `RAVDESS Emotional Speech Audio` |
| **Emotion** (bigger) | CREMA-D (~7.4k clips) | `CREMA-D` |
| **Gender + Age** | Common Voice | `Mozilla Common Voice` |

You'll need a free Kaggle account. Download the zip and extract it anywhere.

### Step 2 — Sort the files into the right folders (one command)

The raw downloads aren't in the folder layout the trainers expect, so use the
included prep scripts to sort them automatically:

```bash
# Emotion from RAVDESS — no sorting needed, the trainer reads filenames:
python train_emotion.py --data_dir path/to/RAVDESS --ravdess --epochs 80

# Gender from RAVDESS (odd actor IDs = male, even = female):
python prepare_ravdess_gender.py --ravdess_dir path/to/RAVDESS
python train_gender.py

# Gender + Age from Common Voice (reads the validated.tsv labels):
python prepare_commonvoice.py --cv_dir path/to/cv-corpus/en --limit 4000
python train_gender.py
python train_age.py
```

### Step 3 — That's it

Each trainer saves its model to `models/`. The next time you launch the app,
the **Engine status** lights turn green and predictions switch from heuristics
to real model output automatically — no code changes needed.

### Want to test the pipeline first (no download)?

Every trainer has a `--demo` flag that trains on synthetic data so you can
confirm the whole flow runs. (These demo models are for pipeline-testing only —
they learn random data, not real voices, so don't use them for real audio.)

```bash
python train_gender.py --demo
python train_age.py --demo
python train_emotion.py --demo
```

### Manual folder layout (if not using Kaggle/prep scripts)

### Manual folder layout (if not using Kaggle/prep scripts)

```bash
python train_gender.py --data_dir datasets/gender --epochs 50
python train_age.py --data_dir datasets/age --epochs 60
python train_emotion.py --data_dir datasets/emotion --epochs 80
```

Each run saves to `models/`:
- `*_model.h5` — the trained network
- `*_encoder.pkl` — label encoder
- `*_scaler.pkl` — feature scaler (gender/age)
- `*_confusion_matrix.png` and `*_training_history.png` — evaluation plots

---

## 🧪 Testing

```bash
# CLI analysis of a single file:
python predict.py path/to/voice.wav
```

This prints gender, age group, emotion, pitch, and speaking rate with
confidence scores and the source (model vs heuristic) of each prediction.

---

## ☁️ Deployment

### Streamlit Community Cloud (free)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick the repo.
3. Set the main file to `app.py`. It installs `requirements.txt` automatically.

> Note: hosted environments usually lack a microphone, so use file upload there.

### Docker (any host)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t voice-ai .
docker run -p 8501:8501 voice-ai
```

---

## ⚡ GPU support

`utils/model_builder.configure_gpu()` is called at the start of every training
script. It auto-detects CUDA GPUs and enables memory growth. For GPU you need:
- An NVIDIA GPU + matching CUDA/cuDNN
- A GPU-enabled TensorFlow build

CPU works fine too — just slower.

---

## 🔮 Future improvements

- **Accent detection** and **language ID** with a pretrained `wav2vec2` model
  from HuggingFace `transformers` (scaffolding noted in `requirements.txt`).
- **Speaker identification / verification** using embeddings (e.g. ECAPA-TDNN).
- **Exact age regression** instead of grouped bands, given a labelled dataset.
- **Data augmentation** (pitch shift, time stretch, noise) to boost accuracy.
- **Real-time streaming** analysis frame-by-frame.
- **Model explainability** (Grad-CAM over the mel spectrogram).

---

## ⚠️ Notes & ethics

Voice-based demographic and emotion predictions are **probabilistic** and can be
biased by accent, recording quality, and dataset imbalance. Use results as
estimates, not ground truth, and be transparent about limitations — especially
for anything affecting real people.

---

## 📜 License

MIT — free to use for academic and portfolio purposes.
