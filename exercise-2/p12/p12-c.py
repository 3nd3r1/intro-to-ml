import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv("toy_train_4096.csv").dropna()

lri_model = LogisticRegression(max_iter=1000)
lri_model.fit(
    np.column_stack([df_train["x1"], df_train["x2"], df_train["x1"] * df_train["x2"]]),
    df_train["y"],
)

intercept = lri_model.intercept_[0]
coef_x1 = lri_model.coef_[0][0]
coef_x2 = lri_model.coef_[0][1]
coef_x1x2 = lri_model.coef_[0][2]

true_intercept = 0.1
true_coef_x1 = -2
true_coef_x2 = 1
true_coef_x1x2 = 0.2

comparison_data = {
    "Coefficient": ["Intercept", "x1", "x2", "x1*x2"],
    "True Model": [true_intercept, true_coef_x1, true_coef_x2, true_coef_x1x2],
    "Estimated (LRi)": [intercept, coef_x1, coef_x2, coef_x1x2],
    "Difference": [
        intercept - true_intercept,
        coef_x1 - true_coef_x1,
        coef_x2 - true_coef_x2,
        coef_x1x2 - true_coef_x1x2,
    ],
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_latex(index=False, float_format="%.4f", escape=True))
