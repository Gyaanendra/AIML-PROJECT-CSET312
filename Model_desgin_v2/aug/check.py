import pandas as pd

df = pd.read_csv('real_120k_t5paws_aug.csv')

print(df.info())

print("="*81)
print(df['aug_type'].value_counts())