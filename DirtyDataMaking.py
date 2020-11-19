import pandas as pd
import random

df = pd.read_csv("train_churn_kg.csv")


df_index = df.columns


for i in range(int(len(df)/3)):
    for j in  range(int(len(df.columns)/3)):
        x = random.randrange(0, len(df))
        y = random.randrange(0, len(df.columns))
 
        df.loc[x,df_index[y]] = 'Null'

print(df)



df.to_csv("train_dirty.csv", header=True, index=False)
