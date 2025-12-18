import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

df_train["class2"] = (df_train["class4"] != "nonevent").astype(int)

features = [
    col
    for col in df_train.columns
    if col not in ["id", "date", "class4", "partlybad", "class2"]
]

df_x_train = df_train[features]
df_y_train = df_train["class2"]
df_x_test = df_test[features]

cv = StratifiedKFold(n_splits=5, shuffle=True)

# Logistic Regression with Ridge
logreg_ridge_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(penalty="l2", C=1.0, max_iter=5000),
        ),
    ]
)

logreg_ridge_scores = cross_val_score(
    logreg_ridge_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)

print(
    f"Logistic Regression (Ridge): {logreg_ridge_scores.mean():.3f} (std {logreg_ridge_scores.std():.3f})"
)

# Logistic Regression with Lasso
logreg_lasso_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000),
        ),
    ]
)

logreg_lasso_scores = cross_val_score(
    logreg_lasso_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)
print(
    f"Logistic Regression (Lasso): {logreg_lasso_scores.mean():.3f} (std {logreg_lasso_scores.std():.3f})"
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf_scores = cross_val_score(rf, df_x_train, df_y_train, cv=cv, scoring="accuracy")
print(f"Random Forest: {rf_scores.mean():.3f} (std {rf_scores.std():.3f})")
