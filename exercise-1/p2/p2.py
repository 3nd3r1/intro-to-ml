import numpy as np
import pandas as pd


df_train = pd.read_csv("./train_syn.csv")
df_valid = pd.read_csv("./valid_syn.csv")
df_test = pd.read_csv("./test_syn.csv")
for deg in range(9):
    pol = np.polynomial.Polynomial.fit(df_train["x"], df_train["y"], deg=deg)
    mean = ((df["y"]-pol(df["x"]))**2).mean()
    print("Deg",deg,":",mean)
