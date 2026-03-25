import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

FILES = [
    "real_120k_lfm_part_4.csv",
    "fake_120k_lfm_part_4.csv",
]

ERROR_PATTERNS = [
    "Failed to connect to Ollama",
    "generation failed: Failed to connect to Ollama",
    "https://ollama.com/download",
    "[generation failed: Failed to connect to Ollama"
]

OUTPUT_SUFFIX_CLEAN = "_clean.csv"
OUTPUT_SUFFIX_BAD   = "_bad_rows.csv"

# ────────────────────────────────────────────────

def is_error_text(text):
    if pd.isna(text):
        return True
    text = str(text).strip()
    if not text:
        return True
    text_lower = text.lower()
    return any(pat.lower() in text_lower for pat in ERROR_PATTERNS)


print("═" * 70)
print(" Checking augmented_text for Ollama generation failures ")
print("═" * 70)

total_rows = 0
total_bad   = 0
total_good  = 0

for original_file in FILES:
    path = Path(original_file)
    if not path.is_file():
        print(f"✗ File not found: {original_file}")
        continue

    print(f"\n→ Processing: {original_file}")

    try:
        df = pd.read_csv(
            original_file,
            dtype=str,
            low_memory=False,
            on_bad_lines='warn'
        )
    except Exception as e:
        print(f"  Error reading file: {e}")
        continue

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    required_cols = {'original_text', 'label', 'tone', 'augmented_text'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  Missing columns: {missing}")
        continue

    total_rows += len(df)

    # Mark bad rows
    df['is_bad'] = df['augmented_text'].apply(is_error_text)

    bad_count  = df['is_bad'].sum()
    good_count = len(df) - bad_count

    total_bad += bad_count
    total_good += good_count

    print(f"  Total rows:      {len(df):,}")
    print(f"  Bad rows:        {bad_count:,}  ({bad_count/len(df)*100:.1f}%)")
    print(f"  Good rows:       {good_count:,}")

    if bad_count == 0:
        print("  → No corrupted rows found")
        continue

    # Show where the bad rows are
    bad_indices = df.index[df['is_bad']].tolist()
    if bad_indices:
        first = bad_indices[0]
        last  = bad_indices[-1]
        print(f"  First bad row:   {first:,}")
        print(f"  Last bad row:    {last:,}")
        print(f"  Span:            {first:,} – {last:,}  ({last - first + 1:,} rows)")

    # ── Save clean version ───────────────────────────────────────
    clean_df = df[~df['is_bad']].copy()
    clean_df = clean_df.drop(columns=['is_bad'], errors='ignore')

    clean_path = path.with_stem(path.stem + OUTPUT_SUFFIX_CLEAN)
    clean_df.to_csv(clean_path, index=False)
    print(f"  Saved clean file → {clean_path.name}  ({len(clean_df):,} rows)")

    # ── Save bad rows only (for possible re-generation) ──────────
    bad_df = df[df['is_bad']].copy()
    bad_df = bad_df.drop(columns=['is_bad'], errors='ignore')

    bad_path = path.with_stem(path.stem + OUTPUT_SUFFIX_BAD)
    bad_df.to_csv(bad_path, index=False)
    print(f"  Saved bad rows   → {bad_path.name}  ({len(bad_df):,} rows)")

# ── Final summary ────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("FINAL SUMMARY")
print("═" * 70)
print(f"Processed rows total:     {total_rows:,}")
print(f"Bad / corrupted rows:     {total_bad:,}  ({total_bad/total_rows*100:.1f}% of all data)")
print(f"Good / usable rows:       {total_good:,}  ({total_good/total_rows*100:.1f}%)")
print()
print(f"Clean files created:")
for f in FILES:
    p = Path(f)
    print(f"  • {p.stem}{OUTPUT_SUFFIX_CLEAN}")
print()
print("You can now:")
print("  • Train on the *_clean.csv files")
print("  • Re-generate only the rows in *_bad_rows.csv")
print("═" * 70)