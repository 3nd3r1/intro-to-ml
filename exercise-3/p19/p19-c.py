import matplotlib.pyplot as plt
import numpy as np
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

# random initialization
losses_random = []
for i in range(1000):
    kmeans = KMeans(
        n_clusters=4, algorithm="lloyd", n_init=1, init="random", random_state=i
    )
    kmeans.fit(df_x_train)
    losses_random.append(kmeans.inertia_)

losses_random = np.array(losses_random)

plt.figure(figsize=(10, 6))
plt.hist(losses_random, bins=50, edgecolor="black")
plt.xlabel("k-means loss")
plt.ylabel("frequency")
plt.title("random")
plt.grid(True, alpha=0.3)
plt.savefig("p19-c-random.png")
plt.show()

min_loss = losses_random.min()
max_loss = losses_random.max()
threshold = min_loss * 1.01
good_losses = np.sum(losses_random <= threshold)
prob_good = good_losses / 1000

print(f"min_loss: {min_loss:.2f}")
print(f"max_loss: {max_loss:.2f}")
print(f"Expected initializations needed: {1/prob_good:.1f}")

# k-means++
losses_kmeanspp = []
for i in range(1000):
    kmeans = KMeans(
        n_clusters=4, algorithm="lloyd", n_init=1, init="k-means++", random_state=i
    )
    kmeans.fit(df_x_train)
    losses_kmeanspp.append(kmeans.inertia_)

losses_kmeanspp = np.array(losses_kmeanspp)

plt.figure(figsize=(10, 6))
plt.hist(losses_kmeanspp, bins=50, edgecolor="black", alpha=0.7, label="k-means++")
plt.hist(losses_random, bins=50, edgecolor="black", alpha=0.5, label="random")
plt.xlabel("k-means loss")
plt.ylabel("frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("k-means++ vs random")
plt.savefig("p19-c-kmpp.png")
plt.show()

min_loss = losses_kmeanspp.min()
max_loss = losses_kmeanspp.max()
threshold = min_loss * 1.01
good_losses = np.sum(losses_kmeanspp <= threshold)
prob_good = good_losses / 1000

print(f"min_loss: {min_loss:.2f}")
print(f"max_loss: {max_loss:.2f}")
print(f"Expected initializations needed: {1/prob_good:.1f}")
