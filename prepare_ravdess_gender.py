"""
prepare_ravdess_gender.py
-------------------------
Builds a GENDER dataset from a RAVDESS download.

RAVDESS filenames look like:
    03-01-05-01-02-01-12.wav
The LAST field (here "12") is the actor ID. By the RAVDESS spec:
    odd actor IDs  = male
    even actor IDs = female

This script walks the RAVDESS folder and COPIES each clip into:
    datasets/gender/male/   or   datasets/gender/female/

so you can then run:
    python train_gender.py

(Use prepare_commonvoice.py instead if you want age too — RAVDESS has no
age labels, only emotion + gender.)

USAGE
-----
    python prepare_ravdess_gender.py --ravdess_dir datasets/RAVDESS

Author: Voice AI System
"""

import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description="RAVDESS -> gender folders")
    parser.add_argument("--ravdess_dir", required=True,
                        help="Folder containing extracted RAVDESS .wav files")
    args = parser.parse_args()

    if not os.path.isdir(args.ravdess_dir):
        raise SystemExit(f"Folder not found: {args.ravdess_dir}")

    os.makedirs("datasets/gender/male", exist_ok=True)
    os.makedirs("datasets/gender/female", exist_ok=True)

    male, female = 0, 0
    for root, _, files in os.walk(args.ravdess_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue
            try:
                actor_id = int(parts[6])
            except ValueError:
                continue

            src = os.path.join(root, fname)
            if actor_id % 2 == 1:   # odd = male
                shutil.copy2(src, f"datasets/gender/male/{fname}")
                male += 1
            else:                    # even = female
                shutil.copy2(src, f"datasets/gender/female/{fname}")
                female += 1

    print(f"Done. male={male}, female={female}")
    print("Next: python train_gender.py")


if __name__ == "__main__":
    main()
