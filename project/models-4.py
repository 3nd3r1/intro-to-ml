import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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

for c in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=5000),
            ),
        ]
    )
    scores = cross_val_score(
        model, df_x_train, df_y_train_binary, cv=cv, scoring="accuracy"
    )
    print(f"C={c}: {scores.mean():.3f} (std {scores.std():.3f})")

base_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)),
    ]
)
base_scores = cross_val_score(
    base_model, df_x_train, df_y_train_binary, cv=cv, scoring="accuracy"
)
print(f"no calibration: {base_scores.mean():.3f} (std {base_scores.std():.3f})")

scaler = StandardScaler()
df_x_scaled = scaler.fit_transform(df_x_train)
base_clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)
calibrated = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
cal_scores = cross_val_score(
    calibrated, df_x_scaled, df_y_train_binary, cv=cv, scoring="accuracy"
)
calibrated_sigmoid = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
cal_sig_scores = cross_val_score(
    calibrated_sigmoid, df_x_scaled, df_y_train_binary, cv=cv, scoring="accuracy"
)
print(
    f"sigmoid calibration: {cal_sig_scores.mean():.3f} (std {cal_sig_scores.std():.3f})"
)

event_mask = df_train["class4"] != "nonevent"
df_x_train_events = df_train.loc[event_mask, features]
df_y_train_events = df_train.loc[event_mask, "class4"]

for c in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=c, max_iter=5000)),
        ]
    )
    scores = cross_val_score(
        model, df_x_train_events, df_y_train_events, cv=cv, scoring="accuracy"
    )
    print(f"C={c}: {scores.mean():.3f} (std {scores.std():.3f})")
