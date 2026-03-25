import pandas as pd

FILES = [
    "real_120k_lfm.csv",
    "real_120k_lfm_part_2.csv",
    "real_120k_lfm_part3.csv",
    "real_120k_lfm_part_4.csv",
    "fake_120k_lfm.csv",
    "fake_120k_lfm_part_2.csv",
    "fake_120k_lfm_part3.csv",
    "fake_120k_lfm_part_4.csv",
]

# Show file options
print("="*60)
print("SELECT A FILE:")
print("="*60)
for i, f in enumerate(FILES):
    print(f"  [{i}] {f}")

# Pick file
file_idx = int(input("\nEnter file number: "))
file = FILES[file_idx]

df = pd.read_csv(f"partial_clean/{file}", engine="python", on_bad_lines="skip")
print(f"\n[OK] Loaded — {df.shape[0]} rows")

# Pick row
row_idx = int(input(f"Enter row number (0 to {df.shape[0]-1}): "))
row = df.iloc[row_idx]

# Display
print("\n" + "="*60)
print(f"FILE  : {file}")
print(f"ROW   : {row_idx}")
print("="*60)
for col, val in row.items():
    print(f"\n[{col}]")
    print(val)

print("\n" + "="*60)