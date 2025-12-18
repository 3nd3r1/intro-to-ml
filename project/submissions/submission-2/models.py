import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

features = [
    col
    for col in df_train.columns
    if col not in ["id", "date", "class4", "partlybad", "class2"]
]

df_x_train = df_train[features]
df_y_train = df_train["class4"]
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
logreg_ridge_y_pred = cross_val_predict(
    logreg_ridge_pipeline, df_x_train, df_y_train, cv=cv
)
print()
print("Logistic Regression (Ridge):")
print(f"Score: {logreg_ridge_scores.mean():.3f} (std {logreg_ridge_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, logreg_ridge_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, logreg_ridge_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))

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
logreg_lasso_y_pred = cross_val_predict(
    logreg_lasso_pipeline, df_x_train, df_y_train, cv=cv
)
print()
print("Logistic Regression (Lasso):")
print(f"Score: {logreg_lasso_scores.mean():.3f} (std {logreg_lasso_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, logreg_lasso_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, logreg_lasso_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf_scores = cross_val_score(rf, df_x_train, df_y_train, cv=cv, scoring="accuracy")
rf_y_pred = cross_val_predict(rf, df_x_train, df_y_train, cv=cv)
print()
print("Random Forest:")
print(f"Score: {rf_scores.mean():.3f} (std {rf_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, rf_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, rf_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))
