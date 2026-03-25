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

for file in FILES:
    print("\n" + "="*60)
    print(f"FILE: {file}")
    print("="*60)

    try:
        df = pd.read_csv(
            f"{file}",
            on_bad_lines="skip",       # fix part3 tokenization errors
            engine="python"
        )

        # Drop source_file column
        if "source_file" in df.columns:
            df.drop(columns=["source_file"], inplace=True)
            print(f"   [INFO] Dropped 'source_file' column")

        # Save cleaned file back
        df.to_csv(f"cleaned/{file}", index=False)

        print(f"   [OK] Saved — {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)}")

    except Exception as e:
        print(f"   [ERROR] {e}")

print("\n" + "="*60)
print("ALL FILES CLEANED & SAVED")
print("="*60)