import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


results = []

plt.subplots(2, 2, figsize=(10, 8))

for i in range(1, 5):
    data = pd.read_csv(f"d{i}.csv")
    model = sm.OLS(data["y"], sm.add_constant(data["x"])).fit()

    plt.subplot(2, 2, i)
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Dataset d{i}.csv")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p5-c.png")
plt.show()
