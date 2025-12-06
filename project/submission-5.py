import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

features = [
    col for col in df_train.columns if col not in ["id", "date", "class4", "partlybad"]
]

df_x_train = df_train[features]
df_x_test = df_test[features]

df_y_train_binary = (df_train["class4"] != "nonevent").astype(int)

scaler = StandardScaler()
df_x_train_scaled = scaler.fit_transform(df_x_train)
df_x_test_scaled = scaler.transform(df_x_test)

base_model = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)
binary_model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
binary_model.fit(df_x_train_scaled, df_y_train_binary)

event_mask = df_train["class4"] != "nonevent"
df_x_train_events = df_train.loc[event_mask, features]
df_y_train_events = df_train.loc[event_mask, "class4"]

event_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000)),
    ]
)
event_model.fit(df_x_train_events, df_y_train_events)

p_event = binary_model.predict_proba(df_x_test_scaled)[:, 1]
binary_pred = (p_event > 0.5).astype(int)
event_pred = event_model.predict(df_x_test)

class4_pred = []
for i in range(len(df_x_test)):
    if binary_pred[i] == 0:
        class4_pred.append("nonevent")
    else:
        class4_pred.append(event_pred[i])

submission = pd.DataFrame({"id": df_test["id"], "class4": class4_pred, "p": p_event})
submission.to_csv("./data/submission5.csv", index=False)

print(submission.head(10))
