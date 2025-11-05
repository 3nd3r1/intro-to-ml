import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df_train = pd.read_csv("./train_syn.csv")


plt.subplots(2, 3)

for i, deg in enumerate([0, 1, 2, 3, 4, 8]):
    pol = np.polynomial.Polynomial.fit(df_train["x"], df_train["y"], deg=deg)

    plt.subplot(2, 3, i + 1)
    plt.scatter(df_train["x"], df_train["y"])
    plt.plot(
        np.linspace(start=-3, stop=3, num=256),
        pol(np.linspace(start=-3, stop=3, num=256)),
        color="red",
        linewidth=2,
        label=f"p={deg}",
    )

    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("p2-b.png")
plt.show()
