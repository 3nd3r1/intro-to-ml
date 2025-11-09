import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def f(x):
    return -2 - x + 0.5 * x**2


results = []
for deg in range(0, 7):
    y_0_values = []
    pred_f_0_values = []
    for _ in range(1000):
        X = np.random.uniform(-3, 3, 10)
        epsilon = np.random.normal(0, 0.4, 10)
        Y = f(X) + epsilon

        model = Pipeline(
            [("poly", PolynomialFeatures(degree=deg)), ("linear", LinearRegression())]
        )
        model.fit(X.reshape(-1, 1), Y)

        y_0 = f(0) + np.random.normal(0, 0.4)
        pred_f_0 = model.predict([[0]])[0]

        y_0_values.append(y_0)
        pred_f_0_values.append(pred_f_0)

    mean_pred_f_0 = np.mean(pred_f_0_values)

    irreducible_error = np.mean((np.array(y_0_values) - f(0)) ** 2)
    bias_squared = (mean_pred_f_0 - f(0)) ** 2
    variance = np.mean((np.array(pred_f_0_values) - mean_pred_f_0) ** 2)
    total = irreducible_error + bias_squared + variance
    mse = np.mean((np.array(y_0_values) - np.array(pred_f_0_values)) ** 2)

    results.append([deg, irreducible_error, bias_squared, variance, total, mse])

df = pd.DataFrame(
    results, columns=["Degree", "Irreducible", "BiasSq", "Variance", "Total", "MSE"]
)

print(df.to_latex(index=False, float_format="%.6f", escape=False))


plt.figure(figsize=(10, 6))

plt.plot(df["Degree"], df["Irreducible"], label="Irreducible")
plt.plot(df["Degree"], df["BiasSq"], label="BiasSq")
plt.plot(df["Degree"], df["Variance"], label="Variance")
plt.plot(df["Degree"], df["MSE"], label="MSE")

plt.ylabel("Value")
plt.xlabel("Degree")
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
