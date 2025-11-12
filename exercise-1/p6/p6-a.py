import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("d2.csv")
w0s = [] # intercepts
w1s = [] # slopes

for _ in range(1000):
    num_samples = len(df[["x"]])
    indices = np.random.choice(num_samples, size=num_samples, replace=True)
    x = df[["x"]].values[indices]
    y = df["y"].values[indices]

    model = LinearRegression()
    model.fit(x, y)
    w0s.append(model.intercept_)
    w1s.append(model.coef_[0])

se_w0 = np.std(w0s, ddof=1)
se_w1 = np.std(w1s, ddof=1)

print(f"se_w0: {se_w0:.6f}")
print(f"se_w1: {se_w1:.6f}")
