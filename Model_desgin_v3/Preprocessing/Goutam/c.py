# clean_failed_rows.py
# Removes failed / rejected / empty rows from both CSV files and overwrites them

import pandas as pd
import os
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────

FILES = [
    "real_120k_lfm_part_2.csv",
    "fake_120k_lfm_part_2.csv",
]

BACKUP_SUFFIX = f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ── Cleaning logic ───────────────────────────────────────────────────────────

def clean_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    print(f"\nProcessing: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        original_count = len(df)
        print(f"  Original rows: {original_count:,}")

        if "augmented_text" not in df.columns:
            print("  Warning: 'augmented_text' column not found → skipping")
            return False

        df["augmented_text"] = df["augmented_text"].fillna("").astype(str).str.strip()

        # Define bad rows
        is_failed = df["augmented_text"].str.contains(r"\[generation failed:", regex=True, na=False)
        is_rejected = df["augmented_text"].str.contains(r"\[rejected:", regex=True, na=False)
        is_empty = df["augmented_text"] == ""

        bad_mask = is_failed | is_rejected | is_empty

        bad_count = bad_mask.sum()
        good_count = original_count - bad_count

        print(f"  Failed rows:     {is_failed.sum():,}")
        print(f"  Rejected rows:   {is_rejected.sum():,}")
        print(f"  Empty rows:      {is_empty.sum():,}")
        print(f"  Total bad rows:  {bad_count:,}")
        print(f"  Good rows:       {good_count:,} ({good_count/original_count*100:.1f}%)")

        if bad_count == 0:
            print("  No bad rows found → file unchanged")
            return True

        # Create backup before overwriting
        backup_path = filepath + BACKUP_SUFFIX
        df.to_csv(backup_path, index=False, encoding="utf-8")
        print(f"  Backup created: {os.path.basename(backup_path)}")

        # Keep only good rows
        cleaned_df = df[~bad_mask].copy()

        # Overwrite original file
        cleaned_df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"  Cleaned file saved back to: {os.path.basename(filepath)}")
        print(f"  → Removed {bad_count:,} rows")

        return True

    except Exception as e:
        print(f"  Error processing {filepath}: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cleaning failed/rejected/empty rows from augmentation CSVs\n")

    success_count = 0
    for f in FILES:
        if clean_file(f):
            success_count += 1

    print("\n" + "─" * 60)
    print(f"Finished. Successfully cleaned {success_count} of {len(FILES)} files.")
    print("Backups were created with suffix .backup_YYYYMMDD_HHMMSS")
    print("You can now continue/resume augmentation — failed rows are gone.")