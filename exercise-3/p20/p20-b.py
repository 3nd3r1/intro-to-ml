import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("train.csv")

df_x_train = pd.DataFrame(
    StandardScaler().fit_transform(
        df_train[[col for col in df_train.columns if col.endswith(".mean")]]
    ),
    columns=[col for col in df_train.columns if col.endswith(".mean")],
    index=df_train.index,
)
df_y_train = df_train["class4"]


pca = PCA().fit(df_x_train)

pve = pca.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pve) + 1), pve, "bo-")
plt.title("PVE")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_pve) + 1), cumulative_pve, "ro-")
plt.title("Cumulative PVE")
plt.grid(True)
plt.tight_layout()

plt.savefig("p20-b-norm.png")
plt.show()

df_x_train_unnorm = df_train[[col for col in df_train.columns if col.endswith(".mean")]]

pca_unnorm = PCA().fit(df_x_train_unnorm)

pve_unnorm = pca_unnorm.explained_variance_ratio_
cumulative_pve_unnorm = np.cumsum(pve_unnorm)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pve) + 1), pve, "bo-", label="Normalized")
plt.plot(range(1, len(pve_unnorm) + 1), pve_unnorm, "ro-", label="Unnormalized")
plt.title("PVE Comparison")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_pve) + 1), cumulative_pve, "bo-", label="Normalized")
plt.plot(
    range(1, len(cumulative_pve_unnorm) + 1),
    cumulative_pve_unnorm,
    "ro-",
    label="Unnormalized",
)
plt.title("Cumulative PVE Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("p20-b-unnorm.png")
plt.show()
