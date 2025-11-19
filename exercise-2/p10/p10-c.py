import numpy as np
import pandas as pd
from scipy.stats import norm


def likelihood(x, class_name):
    means = class_means_train.loc[class_name]
    stds = class_stds_train.loc[class_name]

    likelihoods = norm.pdf(x, loc=means, scale=stds)
    return np.prod(likelihoods)


def posterior_adelie(x):
    p_x_adelie = likelihood(x, "Adelie")
    p_x_notadelie = likelihood(x, "notAdelie")

    up = p_x_adelie * class_probs_train["Adelie"]
    down = up + p_x_notadelie * class_probs_train["notAdelie"]

    return up / down


df_train = pd.read_csv("penguins_train.csv").dropna()
df_test = pd.read_csv("penguins_test.csv").dropna()

features = list(df_train.columns.difference(["species"]))

class_means_train = df_train.groupby("species")[features].mean()
class_stds_train = df_train.groupby("species")[features].std()

class_counts_train = df_train["species"].value_counts()
class_probs_train = (class_counts_train + 1) / (len(df_train) + 2)

posteriors = df_test[features].apply(posterior_adelie, axis=1)
predictions = posteriors.apply(lambda p: "Adelie" if p >= 0.5 else "notAdelie")
accuracy = (predictions == df_test["species"]).mean()

print(f"Accuracy: {accuracy:.4f}")

for i in range(3):
    print(f"Penguin {i+1}: P(Adelie|x) = {posteriors.iloc[i]:.4f}")
