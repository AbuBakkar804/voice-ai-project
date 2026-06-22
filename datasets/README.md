# Datasets

Place your audio data in these folders. Training scripts read the **subfolder
name as the label**.

```
datasets/
├── gender/
│   ├── male/      *.wav
│   └── female/    *.wav
├── age/
│   ├── child/     *.wav
│   ├── teen/      *.wav
│   ├── young/     *.wav
│   └── old/       *.wav
└── emotion/
    ├── happy/     *.wav
    ├── sad/       *.wav
    ├── angry/     *.wav
    ├── neutral/   *.wav
    ├── fear/      *.wav
    └── surprise/  *.wav
```

## Where to download

| Dataset | Use for | Link |
|---|---|---|
| **RAVDESS** | Emotion (also gender) | https://zenodo.org/record/1188976 |
| **CREMA-D** | Emotion | https://github.com/CheyneyComputerScience/CREMA-D |
| **SAVEE** | Emotion | http://kahlan.eps.surrey.ac.uk/savee/ |
| **TESS** | Emotion | https://tspace.library.utoronto.ca/handle/1807/24487 |
| **Common Voice** | Gender / Age (has metadata) | https://commonvoice.mozilla.org/datasets |
| **TIMIT** | Gender / phonetics | https://catalog.ldc.upenn.edu/LDC93S1 |
| **VoxCeleb** | Speaker ID / gender | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ |

## Quick start with RAVDESS (emotion)

RAVDESS encodes the emotion in the filename, so you don't need to sort files
into folders manually:

```bash
# after extracting RAVDESS into datasets/RAVDESS/
python train_emotion.py --data_dir datasets/RAVDESS --ravdess --epochs 80
```

## For Common Voice (gender / age)

Common Voice ships a `validated.tsv` with `gender` and `age` columns. Write a
small script to copy each clip into the matching `gender/` or `age/` subfolder
based on those columns, then run `train_gender.py` / `train_age.py`.

## No data yet?

Every training script supports a `--demo` flag that trains on synthetic data
so you can verify the full pipeline before downloading anything:

```bash
python train_gender.py --demo
python train_age.py --demo
python train_emotion.py --demo
```
