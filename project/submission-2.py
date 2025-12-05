import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression with Ridge
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="saga", C=1.0, max_iter=5000)),
    ]
)

model.fit(df_x_train, df_y_train)

probs = model.predict_proba(df_x_test)
class4_pred = model.predict(df_x_test)

p_event = 1 - probs[:, list(model.classes_).index("nonevent")]

submission = pd.DataFrame({"id": df_test["id"], "class4": class4_pred, "p": p_event})
submission.to_csv("./data/submission2.csv", index=False)

print(submission.head(10))
