import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


df_train = pd.read_csv("./train_syn.csv")
df_valid = pd.read_csv("./valid_syn.csv")
df_test = pd.read_csv("./test_syn.csv")
df_train_and_valid = pd.concat([df_train, df_valid], ignore_index=True)

results = []
for deg in range(9):
    pol_train = PolynomialFeatures(degree=deg)
    model_train = LinearRegression()
    model_train.fit(pol_train.fit_transform(df_train[["x"]]), df_train["y"])

    pol_valid_and_train = PolynomialFeatures(degree=deg)
    model_valid_and_train = LinearRegression()
    model_valid_and_train.fit(
        pol_valid_and_train.fit_transform(
            df_train_and_valid[["x"]],
        ),
        df_train_and_valid["y"],
    )

    mse_train = mean_squared_error(
        df_train["y"], model_train.predict(pol_train.transform(df_train[["x"]]))
    )
    mse_valid = mean_squared_error(
        df_valid["y"], model_train.predict(pol_train.transform(df_valid[["x"]]))
    )
    mse_test = mean_squared_error(
        df_test["y"], model_train.predict(pol_train.transform(df_test[["x"]]))
    )
    mse_test_trva = mean_squared_error(
        df_test["y"],
        model_valid_and_train.predict(pol_valid_and_train.transform(df_test[["x"]])),
    )

    cv_scores = cross_val_score(
        LinearRegression(),
        pol_valid_and_train.transform(
            df_train_and_valid[["x"]],
        ),
        df_train_and_valid["y"],
        cv=10,
        scoring="neg_mean_squared_error",
    )
    mse_cv = -cv_scores.mean()
    results.append([deg, mse_train, mse_valid, mse_test, mse_test_trva, mse_cv])

df_results = pd.DataFrame(
    results,
    columns=["Degree", "Train", "Validation", "Test", "TestTRVA", "CV"],
)

print(df_results.to_latex(index=False, float_format="%.6f", escape=False))
