import os
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────
INPUT_REAL  = "real_120k.csv"
INPUT_FAKE  = "fake_120k.csv"
OUTPUT_DIR  = "./split_parts"
PARTS       = 4
# ─────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_csv(input_path: str, prefix: str):
    print(f"\nSplitting {input_path} ...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"→ total rows: {total_rows:,}")

    chunk_size = (total_rows + PARTS - 1) // PARTS   # ceiling division
    print(f"→ ~{chunk_size:,} rows per part")

    for i in range(PARTS):
        start = i * chunk_size
        end   = min(start + chunk_size, total_rows)
        
        if start >= total_rows:
            break
            
        part_df = df.iloc[start:end]
        out_name = f"{prefix}_part_{i+1}_rows_{start+1}_to_{end}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        part_df.to_csv(out_path, index=False)
        print(f"  → part {i+1:2d}   rows {start+1:6,} – {end:6,}   → {out_name}")

# ── Run ───────────────────────────────────────────────────
split_csv(INPUT_REAL, "real")
split_csv(INPUT_FAKE,  "fake")

print("\nDone. Files are in:", OUTPUT_DIR)
print("You can now copy each part to the corresponding machine.")