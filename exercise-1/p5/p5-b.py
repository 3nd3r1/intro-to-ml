import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


plt.subplots(2, 2, figsize=(10, 8))
for i in range(1, 5):
    data = pd.read_csv(f"d{i}.csv")
    model = sm.OLS(data["y"], sm.add_constant(data["x"])).fit()

    plt.subplot(2, 2, i)
    plt.scatter(data["x"], data["y"], label="Data Points")
    plt.plot(data["x"], model.fittedvalues, color="red", label="Fitted Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Dataset d{i}.csv")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p5-b.png")
plt.show()
