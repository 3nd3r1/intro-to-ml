import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
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

losses = []

for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(df_x_train)
    losses.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), losses, "bo-")
plt.xlabel("number of clusters")
plt.ylabel("k-means loss")
plt.grid(True)
plt.savefig("p19-a.png")
plt.show()
