# sample_extractor_for_cleaning.py
"""
Extracts 6 random examples from each of train/val/test CSV files
and saves them to a JSON file for easy inspection when writing cleaning code.
"""

import pandas as pd
import json
import os
from pathlib import Path
import random

# ────────────────────────────────────────────────
#  CONFIGURATION - change paths if needed
# ────────────────────────────────────────────────
DATA_DIR = Path("./Datasets/non_aug")           # ← adjust to your folder
OUTPUT_JSON = "cleaning_review_samples.json"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "val.csv",
    "test": DATA_DIR / "test.csv",
}

# How many samples per split
SAMPLES_PER_SPLIT = 6

# ────────────────────────────────────────────────
def main():
    all_samples = {}

    for split_name, filepath in FILES.items():
        if not filepath.is_file():
            print(f"❌ File not found: {filepath}")
            continue

        print(f"Reading {split_name} ({filepath}) ...")
        df = pd.read_csv(filepath)

        print(f"  → {len(df):,} rows found")

        if len(df) < SAMPLES_PER_SPLIT:
            print(f"  ⚠️  Fewer than {SAMPLES_PER_SPLIT} rows → taking all")
            selected = df.sample(n=len(df), random_state=42)
        else:
            selected = df.sample(n=SAMPLES_PER_SPLIT, random_state=42)

        # Convert to list of dicts (more JSON friendly)
        records = selected.to_dict(orient="records")

        all_samples[split_name] = records

        print(f"  → Selected {len(records)} examples for {split_name}\n")

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Samples saved to: {OUTPUT_JSON}")
    print("You can now open this file to inspect problematic patterns")
    print("and design your cleaning / filtering / normalization steps.\n")

    # Quick preview in console
    for split, items in all_samples.items():
        print(f"─── {split.upper()} samples ({len(items)}) ───")
        for i, item in enumerate(items, 1):
            text_preview = item.get("text", "")[:120].replace("\n", " ").strip()
            label = item.get("label", "?")
            print(f"  {i}. label={label} | {text_preview}...")
        print()


if __name__ == "__main__":
    main()