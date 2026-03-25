import os
import pandas as pd

FILES = [
    "real_120k_lfm.csv", "real_120k_lfm_part_2.csv",
    "real_120k_lfm_part3.csv", "real_120k_lfm_part_4.csv",
    "fake_120k_lfm.csv", "fake_120k_lfm_part_2.csv",
    "fake_120k_lfm_part3.csv", "fake_120k_lfm_part_4.csv",
]

INPUT_DIR  = "partial_clean"
OUTPUT_DIR = "final"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_final.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

dfs = []

for f in FILES:
    path = os.path.join(INPUT_DIR, f)
    if not os.path.exists(path):
        print(f"[SKIP]  {path}")
        continue

    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    print(f"[READ]  {f:40s} → {len(df):>7,} rows")

    # Concatenate original + augmented into single text column
    def concat_text(row):
        orig = str(row.get("original_text", "") or "").strip()
        aug  = str(row.get("augmented_text", "") or "").strip()
        if orig and aug:
            return orig + " [SEP] " + aug
        return orig or aug

    df["text"]  = df.apply(concat_text, axis=1)
    df["label"] = df["label"].astype(int)

    dfs.append(df[["text", "label"]])

# Combine
combined = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO]  Total rows before clean : {len(combined):,}")

# Drop empty text rows
combined = combined[combined["text"].str.strip() != ""]
combined = combined.dropna(subset=["text", "label"])
print(f"[INFO]  Total rows after clean  : {len(combined):,}")

# Shuffle
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"[INFO]  Rows shuffled")

# Label distribution
dist = combined["label"].value_counts()
print(f"\n[INFO]  Label distribution:")
print(f"        Real (0) : {dist.get(0, 0):>7,}")
print(f"        Fake (1) : {dist.get(1, 0):>7,}")

# Save
combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SAVED] → {OUTPUT_FILE}")
print(f"[DONE]  Shape: {combined.shape}")


# ```

# **What it does:**

# | Step | Detail |
# |---|---|
# | Reads | All 8 CSVs from `partial_clean/` |
# | Combines | `original_text + [SEP] + augmented_text` → single `text` column |
# | Keeps | Only `text` and `label` columns |
# | Cleans | Drops empty/null rows |
# | Shuffles | `sample(frac=1, random_state=42)` — reproducible |
# | Saves | `final/dataset_final.csv` |

# Output will look like:
# ```
# text,label
# "The minister announced... [SEP] Officials confirmed...",0
# "Scientists discover...[SEP] In a groundbreaking...",1
# ...