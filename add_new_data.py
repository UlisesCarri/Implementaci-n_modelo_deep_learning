import pandas as pd

df1 = pd.read_csv("emotion_dataset.csv")
df2 = pd.read_csv("emotions_new.csv")

df_total = pd.concat([df1, df2], ignore_index=True)

# 🔹 Revolver aleatoriamente las filas
df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

df_total.to_csv('emotions_newV2.csv', index=False)
