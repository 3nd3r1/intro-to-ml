import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("./data/train.csv")

features = [
    col for col in df_train.columns if col not in ["id", "date", "class4", "partlybad"]
]

df_x_train = df_train[features]
df_y_train_binary = (df_train["class4"] != "nonevent").astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True)

print("stage 1")
for c in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=10000),
            ),
        ]
    )
    scores = cross_val_score(
        model, df_x_train, df_y_train_binary, cv=cv, scoring="accuracy"
    )
    print(f"C={c}: {scores.mean():.3f} (std {scores.std():.3f})")


event_mask = df_train["class4"] != "nonevent"
df_x_train_events = df_train.loc[event_mask, features]
df_y_train_events = df_train.loc[event_mask, "class4"]

print("stage 2")
for c in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=c, max_iter=10000)),
        ]
    )
    scores = cross_val_score(
        model, df_x_train_events, df_y_train_events, cv=cv, scoring="accuracy"
    )
    print(f"C={c}: {scores.mean():.3f} (std {scores.std():.3f})")
