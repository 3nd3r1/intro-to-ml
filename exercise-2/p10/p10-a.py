import pandas as pd


df_train = pd.read_csv("penguins_train.csv").dropna()
df_test = pd.read_csv("penguins_test.csv").dropna()


stats = df_train.groupby("species").agg(["mean", "std"]).reset_index(drop=True)
print(stats)
