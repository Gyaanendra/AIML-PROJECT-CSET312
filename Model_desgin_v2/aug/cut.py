# cut_to_120k.py
# Run this once to create smaller versions of both files
# Takes only a few minutes

import pandas as pd

# ── Config ────────────────────────────────────────────────
INPUT_FILES = {
    "Real": r"real_300k_matched.csv",      # ← change to your actual path
    "Fake": r"fake_300k_matched.csv",
}

OUTPUT_FILES = {
    "Real": r"real_120k_matched.csv",
    "Fake": r"fake_120k_matched.csv",
}

TARGET_ROWS = 120_000
TEXT_COL = "text"
LABEL_COL = "label"

# ── Process each file ─────────────────────────────────────
for label, in_path in INPUT_FILES.items():
    print(f"\nProcessing {label} ({in_path})")
    
    try:
        df = pd.read_csv(
            in_path,
            dtype={TEXT_COL: 'string', LABEL_COL: 'int'},
            usecols=[TEXT_COL, LABEL_COL],   # faster if you only need these
            low_memory=False
        )
        
        # Remove rows with missing text
        df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
        
        n_original = len(df)
        print(f"  Original rows: {n_original:,}")
        
        if n_original <= TARGET_ROWS:
            print(f"  Already ≤ {TARGET_ROWS:,} → copying as is")
            df.to_csv(OUTPUT_FILES[label], index=False, encoding='utf-8')
            continue
        
        # Take first TARGET_ROWS (or random if you prefer)
        # Option A: first 120k (fastest, deterministic)
        df_small = df.iloc[:TARGET_ROWS].copy()
        
        # Option B: random sample (uncomment if you want variety)
        # df_small = df.sample(n=TARGET_ROWS, random_state=42).reset_index(drop=True)
        
        print(f"  Selected rows: {len(df_small):,}")
        
        # Quick quality check
        print(f"  Missing text : {df_small[TEXT_COL].isna().sum()}")
        print(f"  Unique labels: {df_small[LABEL_COL].value_counts().to_dict()}")
        
        df_small.to_csv(OUTPUT_FILES[label], index=False, encoding='utf-8')
        print(f"  Saved → {OUTPUT_FILES[label]}")
        
    except Exception as e:
        print(f"  Error processing {label}: {e}")

print("\nDone. You can now update your augmentation script to use the new 120k files.")