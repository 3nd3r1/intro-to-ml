import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Task a
df = pd.read_csv("./train.csv")
df.drop(columns=["id", "partlybad"], inplace=True)

# Task b
df_columns = df[["T84.mean", "UV_A.mean", "CS.mean"]]
if not isinstance(df_columns, pd.DataFrame):
    raise TypeError("Something went wrong")
print(df_columns.describe())

# Task c
df_t84_mean = df[["T84.mean"]]
if not isinstance(df_t84_mean, pd.DataFrame):
    raise TypeError("Something went wrong")
array_t84_mean = df_t84_mean.values
if not isinstance(array_t84_mean, np.ndarray):
    raise TypeError("Something went wrong")
print(array_t84_mean.mean())

# Task d
plt.figure()

plt.subplot(121)
x = df["class4"].value_counts().index.values
y = df["class4"].value_counts().values
if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
    raise TypeError("Something went wrong")
plt.bar(x, y)

plt.subplot(122)
x = df["CO242.mean"].values
if not isinstance(x, np.ndarray):
    raise TypeError("Something went wrong")
plt.hist(x)

plt.show()

# Task e
df_columns = df[["UV_A.mean", "T84.mean", "H2O84.mean"]]
if not isinstance(df_columns, pd.DataFrame):
    raise TypeError("Something went wrong")
sns.pairplot(df_columns)

plt.show()

# Task f
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

most_common_class4 = train_df["class4"].mode()[0]
p = (train_df["class4"] != "nonevent").mean()

dummy_predictions_df = pd.DataFrame(
    {"id": test_df["id"], "class4": [most_common_class4] * len(test_df), "p": p}
)
dummy_predictions_df.to_csv("./dummy_predictions.csv", index=False)
