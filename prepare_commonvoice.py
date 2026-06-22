"""
prepare_commonvoice.py
----------------------
Sorts a Mozilla Common Voice download into the folder layout that
train_gender.py and train_age.py expect.

Common Voice ships a TSV (usually `validated.tsv`) with columns including:
    path, gender, age
and the audio clips in a `clips/` folder (usually .mp3).

This script reads the TSV and COPIES each clip into:
    datasets/gender/<male|female>/
    datasets/age/<child|teen|young|old>/

so you can then simply run:
    python train_gender.py
    python train_age.py

USAGE
-----
    python prepare_commonvoice.py \
        --cv_dir  path/to/cv-corpus/en \
        --tsv     validated.tsv \
        --limit   4000        # optional cap per class to keep it balanced

Common Voice age buckets are strings like 'teens', 'twenties', 'thirties',
'fourties', 'fifties', 'sixties', 'seventies', 'eighties'. We map them to
our four groups (child/teen/young/old).

Author: Voice AI System
"""

import os
import csv
import shutil
import argparse
from collections import defaultdict

# Map Common Voice 'age' strings -> our 4 age groups.
AGE_MAP = {
    "teens": "teen",
    "twenties": "young",
    "thirties": "young",
    "fourties": "old",   # CV spells it "fourties"
    "forties": "old",
    "fifties": "old",
    "sixties": "old",
    "seventies": "old",
    "eighties": "old",
}
# Note: Common Voice has very few/zero child samples; if you have a separate
# children's speech set, drop it directly into datasets/age/child/.

GENDER_MAP = {
    "male": "male",
    "female": "female",
    "man": "male",
    "woman": "female",
}


def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice data")
    parser.add_argument("--cv_dir", required=True,
                        help="Folder containing the TSV and clips/ directory")
    parser.add_argument("--tsv", default="validated.tsv",
                        help="TSV filename inside cv_dir")
    parser.add_argument("--clips_subdir", default="clips",
                        help="Subfolder holding the audio clips")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max clips per class (0 = no limit)")
    args = parser.parse_args()

    tsv_path = os.path.join(args.cv_dir, args.tsv)
    clips_dir = os.path.join(args.cv_dir, args.clips_subdir)

    if not os.path.exists(tsv_path):
        raise SystemExit(f"TSV not found: {tsv_path}")
    if not os.path.isdir(clips_dir):
        raise SystemExit(f"Clips folder not found: {clips_dir}")

    # Make destination folders.
    for g in ["male", "female"]:
        os.makedirs(f"datasets/gender/{g}", exist_ok=True)
    for a in ["child", "teen", "young", "old"]:
        os.makedirs(f"datasets/age/{a}", exist_ok=True)

    gender_counts = defaultdict(int)
    age_counts = defaultdict(int)
    copied = 0

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = row.get("path", "").strip()
            if not clip:
                continue
            src = os.path.join(clips_dir, clip)
            if not os.path.exists(src):
                continue

            # ---- gender ----
            g_raw = (row.get("gender") or "").strip().lower()
            g = GENDER_MAP.get(g_raw)
            if g and (args.limit == 0 or gender_counts[g] < args.limit):
                shutil.copy2(src, f"datasets/gender/{g}/{clip}")
                gender_counts[g] += 1

            # ---- age ----
            a_raw = (row.get("age") or "").strip().lower()
            a = AGE_MAP.get(a_raw)
            if a and (args.limit == 0 or age_counts[a] < args.limit):
                shutil.copy2(src, f"datasets/age/{a}/{clip}")
                age_counts[a] += 1

            copied += 1
            if copied % 1000 == 0:
                print(f"  processed {copied} rows...")

    print("\nDone.")
    print("Gender distribution:", dict(gender_counts))
    print("Age distribution   :", dict(age_counts))
    print("\nNext steps:")
    print("  python train_gender.py")
    print("  python train_age.py")
    if age_counts.get("child", 0) == 0:
        print("\nNote: Common Voice has no 'child' clips. Add child speech")
        print("manually to datasets/age/child/ for a 4-class age model,")
        print("or train on the 3 available groups.")


if __name__ == "__main__":
    main()
