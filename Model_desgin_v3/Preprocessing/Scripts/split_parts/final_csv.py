import  pandas as pd

df = pd.read_csv("final/dataset_final.csv")
print(f"Total rows: {len(df):,}")
print("\nLabel distribution:")
print(df["label"].value_counts())