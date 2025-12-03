import matplotlib.pyplot as plt
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


pca = PCA(n_components=2)
x_pca = pca.fit_transform(df_x_train)

plt.figure(figsize=(10, 8))
classes = df_y_train.unique()
markers = ["o", "s", "^", "D"]
colors = ["blue", "red", "green", "orange"]

for i, cls in enumerate(classes):
    mask = df_y_train == cls
    plt.scatter(
        x_pca[mask, 0],
        x_pca[mask, 1],
        c=colors[i],
        marker=markers[i],
        label=cls,
        alpha=0.6,
        s=50,
    )

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("p20-a.png")
plt.show()
