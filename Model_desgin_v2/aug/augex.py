import pandas as pd 

df = pd.read_csv('real_aug_hf_newprompt.csv')


print(df['aug_type'].value_counts())