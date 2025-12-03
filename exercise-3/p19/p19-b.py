import pandas as pd
from scipy.optimize import linear_sum_assignment
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

kmeans = KMeans(n_clusters=4, algorithm="lloyd", n_init=10)
cluster_labels = kmeans.fit_predict(df_x_train)

tt = pd.DataFrame({"class": df_y_train, "cluster": cluster_labels})
tt = tt.groupby(["class", "cluster"]).size().unstack(fill_value=0)

lsa = linear_sum_assignment(tt, maximize=True)
pairs = {tt.index[i]: tt.columns[j] for i, j in zip(*lsa)}

tt_reordered = tt.loc[list(pairs.keys()), list(pairs.values())]

print(tt_reordered.to_latex())
