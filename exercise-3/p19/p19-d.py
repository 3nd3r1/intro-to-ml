import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
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


z_single = linkage(df_x_train, method="single")
z_complete = linkage(df_x_train, method="complete")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
dendrogram(z_single, no_labels=True)
plt.title("single linkage")

plt.subplot(1, 2, 2)
dendrogram(z_complete, no_labels=True)
plt.title("complete linkage")
plt.tight_layout()

plt.savefig("p19-d.png")
plt.show()

clusters_single = fcluster(z_single, t=4, criterion="maxclust")
clusters_complete = fcluster(z_complete, t=4, criterion="maxclust")

tt_single = pd.DataFrame({"class": df_y_train, "cluster": clusters_single})
tt_single = tt_single.groupby(["class", "cluster"]).size().unstack(fill_value=0)

tt_complete = pd.DataFrame({"class": df_y_train, "cluster": clusters_complete})
tt_complete = tt_complete.groupby(["class", "cluster"]).size().unstack(fill_value=0)

print(tt_single.to_latex())
print(tt_complete.to_latex())
