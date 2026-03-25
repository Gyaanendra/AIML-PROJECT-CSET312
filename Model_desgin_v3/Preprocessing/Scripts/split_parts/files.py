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
        df = pd.read_csv(f"cleaned/{file}")
        
        print(f"\n>>> SHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
        
        print(f"\n>>> COLUMNS ({len(df.columns)}):")
        for col in df.columns:
            print(f"   - {col}")
        
        print(f"\n>>> INFO:")
        df.info()
        
    except FileNotFoundError:
        print(f"   [ERROR] File not found: {file}")
    except Exception as e:
        print(f"   [ERROR] {e}")

print("\n" + "="*60)
print("EXPLORATION COMPLETE")
print("="*60)