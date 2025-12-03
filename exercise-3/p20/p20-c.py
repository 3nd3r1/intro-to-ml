import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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


df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df_x_train, df_y_train, test_size=0.5
)

clf = LogisticRegression(max_iter=1000).fit(df_x_train, df_y_train)

pred_y = clf.predict(df_x_test)
acc_no_pca = accuracy_score(df_y_test, pred_y)

print(f"accuracy without PCA: {acc_no_pca:.4f}")

df_x_all = pd.concat([df_x_train, df_x_test], axis=0)

pca = PCA().fit(df_x_all)

accuracies = []

for n in range(1, 51):
    df_x_train_pca = pca.transform(df_x_train)[:, :n]
    df_x_test_pca = pca.transform(df_x_test)[:, :n]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(df_x_train_pca, df_y_train)

    pred_y = clf.predict(df_x_test_pca)
    acc = accuracy_score(df_y_test, pred_y)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), accuracies, "bo-")
plt.axhline(y=acc_no_pca, color="r", linestyle="--", label=f"No PCA: {acc_no_pca:.4f}")
plt.xlabel("PCs")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.savefig("p20-c.png")
plt.show()

optimal_n = range(1, 51)[np.argmax(accuracies)]
optimal_acc = max(accuracies)
print(f"optimal: {optimal_n} components")
print(f"best test accuracy: {optimal_acc:.4f}")
