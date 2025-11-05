import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR


df_train = pd.read_csv("./train_real.csv")
df_test = pd.read_csv("./test_real.csv")

print(df_train)

models = {
    "Dummy": DummyRegressor(strategy="mean"),
    "OLS": LinearRegression(),
    "RF": RandomForestRegressor(),
    "SVR": SVR(),
    "Ridge": Ridge(),
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(
        model,
        df_train.drop(columns=["Next_Tmax"]),
        df_train["Next_Tmax"],
        cv=10,
        scoring="neg_mean_squared_error",
    )
    model.fit(df_train.drop(columns=["Next_Tmax"]), df_train["Next_Tmax"])

    rmse_train = np.sqrt(
        mean_squared_error(
            df_train["Next_Tmax"],
            model.predict(df_train.drop(columns=["Next_Tmax"])),
        )
    )
    rmse_test = np.sqrt(
        mean_squared_error(
            df_test["Next_Tmax"],
            model.predict(df_test.drop(columns=["Next_Tmax"])),
        )
    )
    rmse_cv = np.sqrt(-cv_scores.mean())

    results.append([name, rmse_train, rmse_test, rmse_cv])

df_results = pd.DataFrame(results, columns=["Regressor", "Train", "Test", "CV"])

print(df_results.to_latex(index=False, float_format="%.6f", escape=False))
