# download_and_filter_120_rows.py
import kagglehub
import pandas as pd
import os
from pathlib import Path

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

DATASET_SLUG = "gyaanendraprakash/aiml-project-cset312"

# Expected filenames in the dataset (adjust if names are slightly different)
# After (correct - with spaces)
REAL_FILE    = "real_300k_matched.csv"
FAKE_FILE    = "fake_300k_matched.csv"

TEXT_COLUMN  = "text"           # change if your text column has different name
OUTPUT_REAL  = "real_120k.csv"
OUTPUT_FAKE  = "fake_120k.csv"

MIN_WORDS    = 25
MAX_WORDS    = 350
KEEP_ROWS    = 120_000
# ────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Simple word count – splits on whitespace"""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(text.strip().split())


def filter_and_sample(df: pd.DataFrame, n_rows: int = KEEP_ROWS) -> pd.DataFrame:
    """
    1. Calculate word count
    2. Filter rows where word count is between MIN_WORDS and MAX_WORDS
    3. Take first n_rows (or all if fewer qualify)
    """
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in the DataFrame")

    print(f"   Original rows: {len(df):,}")

    df = df.copy()
    df['word_count'] = df[TEXT_COLUMN].apply(count_words)

    mask = (df['word_count'] >= MIN_WORDS) & (df['word_count'] <= MAX_WORDS)
    filtered = df[mask].drop(columns=['word_count'])

    print(f"   Rows after word count filter ({MIN_WORDS}–{MAX_WORDS}): {len(filtered):,}")

    if len(filtered) == 0:
        raise ValueError("No rows remain after filtering by word count")

    # Take first KEEP_ROWS (or shuffle if you prefer random sample)
    result = filtered.head(n_rows).reset_index(drop=True)

    print(f"   Final selected rows: {len(result)}")
    return result


def main():
    print("Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download(DATASET_SLUG)
        print("Dataset downloaded to:", dataset_path)
    except Exception as e:
        print("Error downloading dataset:", e)
        print("Make sure you have logged in with kagglehub.login() or have ~/.kaggle/kaggle.json")
        return

    dataset_dir = Path(dataset_path)

    # Find the actual files
    real_path = dataset_dir / REAL_FILE
    fake_path = dataset_dir / FAKE_FILE

    if not real_path.exists():
        print(f"File not found: {real_path}")
        print("Files in dataset folder:")
        for f in dataset_dir.glob("*"):
            print("   ", f.name)
        return

    if not fake_path.exists():
        print(f"File not found: {fake_path}")
        return

    # ── Process real ───────────────────────────────────────
    print("\nProcessing real news...")
    df_real = pd.read_csv(real_path, low_memory=False)
    df_real_small = filter_and_sample(df_real)
    df_real_small.to_csv(OUTPUT_REAL, index=False, encoding="utf-8")
    print(f"Saved → {OUTPUT_REAL} ({len(df_real_small)} rows)")

    # ── Process fake ───────────────────────────────────────
    print("\nProcessing fake news...")
    df_fake = pd.read_csv(fake_path, low_memory=False)
    df_fake_small = filter_and_sample(df_fake)
    df_fake_small.to_csv(OUTPUT_FAKE, index=False, encoding="utf-8")
    print(f"Saved → {OUTPUT_FAKE} ({len(df_fake_small)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()