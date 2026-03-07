import pandas as pd 

df = pd.read_csv('real_120k_t5paws_aug.csv')

df_og = pd.read_csv('real_120k_matched.csv')


print(df['text'][111])
print("="*81)
print(df_og['text'][111])

