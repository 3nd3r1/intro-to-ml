# Preliminary Report

**Team Name:** Viljami Ranta
**Team Members:** Viljami Ranta

## 1. Introduction

We are gonna do this!

## 2. Data Exploration

### 2.1 Dataset Overview

Let's start by loading the data:

```python
import pandas as pd
import numpy as np


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
```

Let's look at some basic information:

```python
print(f"taining samples: {len(df_train)}")
print(f"test samples: {len(df_test)}")
print(f"number of features: {df_train.shape[1]}")
print(f"first rows: \n{df_train.head()}")
```

```text
taining samples: 450
test samples: 965
number of features: 104
first rows:
   id        date    class4  partlybad  CO2168.mean  CO2168.std  CO2336.mean  ...   T84.std  UV_A.mean   UV_A.std  UV_B.mean  UV_B.std   CS.mean    CS.std
0   0  2000-03-21        II      False   372.396757    0.752494   372.279392  ...  2.283825   6.237543   4.372063   0.115203  0.104295  0.000510  0.000123
1   1  2000-03-23  nonevent      False   372.889867    0.410639   372.769205  ...  1.979027  11.626868   7.208083   0.301720  0.229672  0.000706  0.000250
2   2  2000-04-07        Ia      False   373.869464    0.655604   373.788580  ...  1.929516  16.688892  10.504951   0.561251  0.451130  0.000851  0.000244
3   3  2000-04-09        Ib      False   376.006588    1.109789   375.888889  ...  3.161601  17.456796  10.967471   0.716453  0.572409  0.002083  0.000203
4   4  2000-04-14  nonevent      False   374.068239    1.257096   374.042330  ...  0.929537   4.279844   2.425409   0.146308  0.106017  0.002650  0.000891

[5 rows x 104 columns]
```

The training set contains 450 samples and the test set 965 samples, with 104 features.

After removing id, date, class4, and partlybad, we have approximately 100 numeric features to work with.

The test set is roughly twice the size of the training set.
This means our model needs to generalize well from pretty small training samples.

Looking at the features, we can identify several groups:

- Temperature (T48, T672, etc.) at various heights
- CO2 concentration (CO2168, CO2336) at different heights
- UV radiation (UV_A, UV_B)
- Condensation sink (CS) - a key variable for particle formation physics

Each measurement appears twice as .mean and once as .std, giving us both the average value and the variability in a day.
The std features might capture if conditions were stable, this may be relevant since particle formation might require stable conditions?

Partlybad likely flags days with data quality issues?

### 2.2 Class Distribution

Let's look at the class distribution:

```python
print("class4 distribution:")
print(df_train["class4"].value_counts())
print("class4 proportion:")
print(df_train["class4"].value_counts(normalize=True).round(3))
```

```text
class4 distribution:
class4
nonevent    225
II          117
Ib           82
Ia           26
Name: count, dtype: int64
class4 proportion:
class4
nonevent    0.500
II          0.260
Ib          0.182
Ia          0.058
Name: proportion, dtype: float64
```

The binary classes are balanced: 225 event and 225 nonevent days.

Multi-class distribution is imbalanced.
Class II is most common (117 samples, 52%), followed by Ib (82 samples, 36%), and Ia (26 samples, 12%).
With only 12% Ia the model will probably struggle to learn this category.

The approach will probably be to focus more on the binary classification then consider multi-class.

### 2.3 Feature Statistics

Let's look at the statictics of the features.
I have only included one representative feature from each "group" we identified in 2.1.

```python
example_features = [
    "CO2168.mean",
    "CO2168.std",
    "Glob.mean",
    "Glob.std",
    "H2O168.mean",
    "H2O168.std",
    "NET.mean",
    "NET.std",
    "NO168.mean",
    "NO168.std",
    "O3168.mean",
    "O3168.std",
    "Pamb0.mean",
    "Pamb0.std",
    "PAR.mean",
    "PAR.std",
    "PTG.mean",
    "PTG.std",
    "RGlob.mean",
    "RGlob.std",
    "RHIRGA168.mean",
    "RHIRGA168.std",
    "RPAR.mean",
    "RPAR.std",
    "SO2168.mean",
    "SO2168.std",
    "SWS.mean",
    "SWS.std",
    "T168.mean",
    "T168.std",
    "UV_A.mean",
    "UV_A.std",
    "CS.mean",
    "CS.std",
]

print("summary of features:")
print(df_train[example_features].describe().T)
```

```text
summary of features:
                count        mean         std         min         25%         50%         75%          max
CO2168.mean     450.0  382.077392   11.168050  359.086782  373.872136  381.358306  389.397318   414.863871
CO2168.std      450.0    3.352010    3.448155    0.120513    0.974968    2.228357    4.529128    22.822280
Glob.mean       450.0  191.993193  126.612076    3.926206   65.420488  199.001375  298.567350   426.457432
Glob.std        450.0  141.429135   92.282796    2.006352   41.814270  157.012442  222.717628   318.331187
H2O168.mean     450.0    7.172941    3.867862    1.001045    3.937315    6.512855    9.613407    19.147809
H2O168.std      450.0    0.568149    0.508993    0.013666    0.212045    0.433215    0.770422     3.871835
NET.mean        450.0  121.420757   87.461837  -59.714025   38.391981  122.180687  196.112464   302.605974
NET.std         450.0  123.781084   82.799235    1.634892   36.537517  130.578440  195.967998   276.463693
NO168.mean      450.0    0.105455    0.334706   -0.015238    0.017327    0.037334    0.081665     5.111429
NO168.std       450.0    0.105686    0.149069    0.020957    0.050170    0.062787    0.093462     1.652820
O3168.mean      450.0   33.333048   10.057395    0.954429   26.775829   33.126153   40.556641    71.332135
O3168.std       450.0    3.639520    2.483059    0.210093    1.719552    3.203109    4.952171    14.467550
Pamb0.mean      450.0  991.144432   11.057177  954.884227  983.727635  991.672481  998.016288  1022.559463
Pamb0.std       450.0    1.092051    0.860455    0.069785    0.426963    0.869860    1.451730     5.524911
PAR.mean        450.0  379.375659  248.637376    9.967239  130.642704  396.158869  595.231906   825.891818
PAR.std         450.0  279.864042  182.989624    4.051870   86.542291  306.228276  440.495389   601.127764
PTG.mean        450.0    0.000531    0.008530   -0.008146   -0.002441   -0.000150    0.000933     0.091026
PTG.std         450.0    0.009061    0.006988    0.000000    0.003923    0.008215    0.012879     0.051675
RGlob.mean      450.0   27.438963   17.313305  -16.464558   11.337135   28.441062   41.784095   102.053937
RGlob.std       450.0   18.385126   10.210336    0.557375    8.560394   21.464233   26.493845    48.313840
RHIRGA168.mean  450.0   69.340439   19.768868   29.028571   53.731021   68.497712   88.377996   115.148868
RHIRGA168.std   450.0    8.337183    5.752477    0.286678    2.339119    8.746017   12.425134    24.642558
RPAR.mean       450.0   19.657715   14.972412    0.028685    9.055261   18.327426   25.807477   148.944451
RPAR.std        450.0   13.740073    9.412204    0.212517    7.349019   13.953458   17.221797    76.968313
SO2168.mean     450.0    0.278566    0.429874   -0.027429    0.055506    0.125699    0.326744     4.263514
SO2168.std      450.0    0.164050    0.146433    0.044047    0.076425    0.110288    0.197420     1.104045
SWS.mean        450.0  906.997929   34.559476  711.615385  909.669292  918.322511  922.294516   932.454545
SWS.std         450.0   23.477093   41.933451    0.000000    0.829277    2.062442   22.131576   190.651578
T168.mean       450.0    6.355534    9.816446  -23.618087   -0.836662    7.560464   13.985403    26.703431
T168.std        450.0    1.833917    1.126904    0.041324    0.814760    1.904278    2.726529     5.246012
UV_A.mean       450.0   10.797330    6.626678    0.438965    4.304971   11.506937   16.573188    22.500037
UV_A.std        450.0    7.613890    4.946876    0.121475    2.700430    8.017732   11.832019    16.320440
CS.mean         450.0    0.003083    0.002246    0.000343    0.001436    0.002548    0.004119     0.013706
CS.std          450.0    0.000692    0.000678    0.000023    0.000290    0.000525    0.000825     0.006277
```

Features have very different scales.
CO2 average at 382, temperature (T168) around 6, condensation sink (CS) around 0.003.
Standardization will be necessary for algorithms like SVM and logistic regression.

Some interesting properties:

- High variability: Glob and PAR have very large STD, indicating highly variable conditions across days
- Near-zero values: PTG.mean is around 0.0005
- Negatives: NET.mean and RGlob.mean can go negative
- Temperature range: T168.mean spans from -24 to +27, capturing the whole climate

### 2.4 Feature Correlations

Let's look at the correlations between features.
I will separately check mean and std features.
I will only check the example_features (one per "group") because features in the same group are highly correlated.

```python
example_features_mean = [feature for feature in example_features if ".mean" in feature]
example_features_std = [feature for feature in example_features if ".std" in feature]
correlation_matrix_mean = df_train[example_features_mean].corr()
correlation_matrix_std = df_train[example_features_std].corr()

high_correlation_pairs = []
for i in range(len(correlation_matrix_mean.columns)):
    for j in range(i + 1, len(correlation_matrix_mean.columns)):
        if abs(correlation_matrix_mean.iloc[i, j]) > 0.9:
            high_correlation_pairs.append(
                {
                    "feature1": correlation_matrix_mean.columns[i],
                    "feature2": correlation_matrix_mean.columns[j],
                    "correlation": correlation_matrix_mean.iloc[i, j],
                }
            )
for i in range(len(correlation_matrix_std.columns)):
    for j in range(i + 1, len(correlation_matrix_std.columns)):
        if abs(correlation_matrix_std.iloc[i, j]) > 0.9:
            high_correlation_pairs.append(
                {
                    "feature1": correlation_matrix_std.columns[i],
                    "feature2": correlation_matrix_std.columns[j],
                    "correlation": correlation_matrix_std.iloc[i, j],
                }
            )

print("highly correlated feature pairs:")
for pair in high_correlation_pairs:
    print(f"{pair['feature1']} and {pair['feature2']}: {pair['correlation']:.3f}")
```

```text
highly correlated feature pairs:
Glob.mean and NET.mean: 0.959
Glob.mean and PAR.mean: 0.996
Glob.mean and RHIRGA168.mean: -0.907
Glob.mean and UV_A.mean: 0.985
NET.mean and PAR.mean: 0.968
NET.mean and UV_A.mean: 0.968
PAR.mean and RHIRGA168.mean: -0.912
PAR.mean and UV_A.mean: 0.993
Glob.std and NET.std: 0.983
Glob.std and PAR.std: 0.997
Glob.std and UV_A.std: 0.988
NET.std and PAR.std: 0.984
NET.std and UV_A.std: 0.978
PAR.std and UV_A.std: 0.993
```

The radiation-related features (Glob, PAR, UV_A, NET) are all highly correlated with each other.
This makes sense, they all basically measure solar radiation.

Relative humidity (RHIRGA168) shows strong negative correlation with radiation variables.
Sunny days are drier, very intuitive.

For modeling, this likely means we don't need to use all of them.

### 2.5 Feature Distributions by Class

For this let's create a column class2 that is event if class4 is nonevent, and nonevent otherwise:

```python
df_train["class2"] = df_train["class4"].apply(lambda x: "nonevent" if x == "nonevent" else "event")
```

Now let's look at the distribution of features by class2:

```python
class2_means = df_train.groupby("class2")[example_features].mean()
class2_mean_diff = abs(class2_means.loc["event"] - class2_means.loc["nonevent"])
class2_mean_diff_normalized = class2_mean_diff / df_train[example_features].std()
class2_top_features = class2_mean_diff_normalized.sort_values(ascending=False)

print("top features by normalized mean different between classes (class2):")
for feature, diff in class2_top_features.items():
    print(f"{feature}: {diff:.3f}")
```

```text
top features by normalized mean different between classes (class2):
RHIRGA168.mean: 1.359
Glob.mean: 1.268
PAR.mean: 1.245
UV_A.mean: 1.186
NET.std: 1.160
Glob.std: 1.158
NET.mean: 1.148
RHIRGA168.std: 1.148
PAR.std: 1.137
RGlob.mean: 1.088
UV_A.std: 1.086
T168.std: 1.058
RGlob.std: 1.051
O3168.mean: 0.905
PTG.std: 0.814
SWS.mean: 0.628
CO2168.mean: 0.570
RPAR.mean: 0.570
RPAR.std: 0.559
CS.mean: 0.484
SWS.std: 0.484
PTG.mean: 0.432
H2O168.mean: 0.425
T168.mean: 0.375
Pamb0.std: 0.375
Pamb0.mean: 0.371
NO168.mean: 0.318
H2O168.std: 0.267
SO2168.mean: 0.260
O3168.std: 0.249
NO168.std: 0.220
CO2168.std: 0.179
SO2168.std: 0.062
CS.std: 0.048
```

Strong predictors:

- Relative humidity (RHIRGA168) and radiation features (Glob, PAR, UV_A, NET) show strong separation
- Temperature variability (T168.std) also show strong separation
- Event days appear to have lower humidity and higher radiation (sunny days)

Moderate predictors:

- Ozone and PTG variability show reasonable separation
- CO2 and reflected radiation (RPAR, RGlob) may contribute in combination with other features

Weak predictors:

- Condensation sink (CS.mean), temperature mean (T168.mean), pressure (Pamb0) and NO and SO2 show limited separation on their own
- CS.std has almost no separation despite being pretty relevant with weather?

The std features rank differently than the same feature means.
This maybe suggests that stable vs fluctuating conditions matter more than the measurements themselves.

## 3. Data Preprocessing

### 3.1 Target

We added the binary target class2 column that is nonevent if class4 is nonevent, and event otherwise.

### 3.2 Scaling

Scaling for scale-sensitive models (logistic regression, SVM).
Tree-based models (Random Forest, Gradient Boosting) don't require scaling.

### 3.3 Feature Selection

Nothing at this point.

We found redundancy in radiation features (Glob, PAR, UV_A, NET) and within same "groups" at different heights.
However, we will let models handle this (Lasso, Ridge).

## 4. Modeling

### 4.1 Validation

We use stratified 5-fold cross-validation to evaluate models.

With 450 training samples and 5 folds:

- Each fold uses 360 samples for training 90 for validation
- We report mean accuracy and standard deviation

### 4.2 Models

#### Model 1: Logistic Regression

- Good baseline and interpretability
- Handles reduntant features (Ridge or Lasso)
- Needs feature scaling

#### Model 2: Random Forest

- Handles reduntant features
- Good baseline and interpretability
- No need for feature scaling
- Provides feature importance

### 4.3 Validation Strategy

We will use k-fold cross-validation with 5 folds.

## 5. Submissions

### 5.1 Submission 1

#### 5.1.1 Approach

For the first submission we only predicted the binary class2.

The hyperparameters:

- C = 1.0 balanced level of regularization
- solver = saga works well with many features and required for Lasso
- max_iter = 5000 to ensure converging
- we score with accuracy

We tested three models for binary classification using 5-fold cross-validation:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True)

features = [
    col
    for col in df_train.columns
    if col not in ["id", "date", "class4", "partlybad", "class2"]
]

df_x_train = df_train[features]
df_y_train = df_train["class2"]
df_x_test = df_test[features]

logreg_ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000)),
])
logreg_ridge_scores = cross_val_score(
    logreg_ridge_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)
print(
    f"Logistic Regression (Ridge): {logreg_ridge_scores.mean():.3f} (std {logreg_ridge_scores.std():.3f})"
)

logreg_lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)),
])
logreg_lasso_scores = cross_val_score(
    logreg_lasso_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)
print(
    f"Logistic Regression (Lasso): {logreg_lasso_scores.mean():.3f} (std {logreg_lasso_scores.std():.3f})"
)

rf = RandomForestClassifier(n_estimators=100)
rf_scores = cross_val_score(rf, df_x_train, df_y_train, cv=cv, scoring="accuracy")

print(f"Random Forest: {rf_scores.mean():.3f} (std {rf_scores.std():.3f})")
```

```text
Logistic Regression (Ridge): 0.860 (std 0.044)
Logistic Regression (Lasso): 0.884 (std 0.026)
Random Forest: 0.860 (std 0.030)
```

LR Lasso performed best, probably because it zeroed the redundant features.
Random Forest underperformed, maybe due to the small dataset.

#### 5.1.2 Submission

Based on cv results, we select LR with Lasso.
Since our model only does binary classification,
we predict "II" for all event days since it's the most common.

```python
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)),
    ]
)

model.fit(df_x_train, df_y_train)

probs = model.predict_proba(df_x_test)[:, 1]

class4_pred = np.where(probs > 0.5, "II", "nonevent")

submission = pd.DataFrame({"id": df_test["id"], "class4": class4_pred, "p": probs})
submission.to_csv("./data/submission1.csv", index=False)
```

#### 5.1.3 Result

Kaggle Score: 0.75712

Binary classification works well, but always predicting "II" probably hurts the multi-class accuracy.

### 5.2 Submission 2

#### 5.2.1 Approach

To improve the score let's try to predict all four classes with the same models.

The hyperparameters remained the same as Submission #1:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True)

features = [
    col
    for col in df_train.columns
    if col not in ["id", "date", "class4", "partlybad", "class2"]
]

df_x_train = df_train[features]
df_y_train = df_train["class4"]
df_x_test = df_test[features]

logreg_ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000)),
])

logreg_lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)),
])

rf = RandomForestClassifier(n_estimators=100)

logreg_ridge_scores = cross_val_score(
    logreg_ridge_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)
logreg_lasso_scores = cross_val_score(
    logreg_lasso_pipeline, df_x_train, df_y_train, cv=cv, scoring="accuracy"
)
rf_scores = cross_val_score(rf, df_x_train, df_y_train, cv=cv, scoring="accuracy")

print(
    f"Logistic Regression (Ridge): {logreg_ridge_scores.mean():.3f} (std {logreg_ridge_scores.std():.3f})"
)
print(
    f"Logistic Regression (Lasso): {logreg_lasso_scores.mean():.3f} (std {logreg_lasso_scores.std():.3f})"
)
print(f"Random Forest: {rf_scores.mean():.3f} (std {rf_scores.std():.3f})")
```

```text
Logistic Regression (Ridge): 0.649 (std 0.023)
Logistic Regression (Lasso): 0.660 (std 0.029)
Random Forest: 0.656 (std 0.032)
```

As expected, multi-class is alot harder than binary.
Let's look more closely at class performance to see if we can do something.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

logreg_ridge_y_pred = cross_val_predict(
    logreg_ridge_pipeline, df_x_train, df_y_train, cv=cv
)
logreg_lasso_y_pred = cross_val_predict(
    logreg_lasso_pipeline, df_x_train, df_y_train, cv=cv
)
rf_y_pred = cross_val_predict(rf, df_x_train, df_y_train, cv=cv)

print("Logistic Regression (Ridge):")
print(f"Score: {logreg_ridge_scores.mean():.3f} (std {logreg_ridge_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, logreg_ridge_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, logreg_ridge_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))

print()
print("Logistic Regression (Lasso):")
print(f"Score: {logreg_lasso_scores.mean():.3f} (std {logreg_lasso_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, logreg_lasso_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, logreg_lasso_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))

print()
print("Random Forest:")
print(f"Score: {rf_scores.mean():.3f} (std {rf_scores.std():.3f})")
print("Classification report:")
print(classification_report(df_y_train, rf_y_pred))
print("Confusion matrix:")
print(confusion_matrix(df_y_train, rf_y_pred, labels=["Ia", "Ib", "II", "nonevent"]))
```

```text
Logistic Regression (Ridge):
Score: 0.651 (std 0.041)
Classification report:
              precision    recall  f1-score   support

          II       0.51      0.52      0.51       117
          Ia       0.22      0.19      0.20        26
          Ib       0.44      0.40      0.42        82
    nonevent       0.85      0.88      0.87       225

    accuracy                           0.66       450
   macro avg       0.50      0.50      0.50       450
weighted avg       0.65      0.66      0.66       450

Confusion matrix:
[[  5  12   5   4]
 [  8  33  34   7]
 [  7  26  61  23]
 [  3   4  20 198]]

Logistic Regression (Lasso):
Score: 0.656 (std 0.020)
Classification report:
              precision    recall  f1-score   support

          II       0.49      0.59      0.53       117
          Ia       0.25      0.19      0.22        26
          Ib       0.42      0.29      0.35        82
    nonevent       0.87      0.89      0.88       225

    accuracy                           0.66       450
   macro avg       0.51      0.49      0.49       450
weighted avg       0.65      0.66      0.65       450

Confusion matrix:
[[  5   9   8   4]
 [  5  24  46   7]
 [  7  21  69  20]
 [  3   3  19 200]]

Random Forest:
Score: 0.647 (std 0.008)
Classification report:
              precision    recall  f1-score   support

          II       0.47      0.50      0.48       117
          Ia       0.20      0.04      0.06        26
          Ib       0.38      0.37      0.37        82
    nonevent       0.83      0.90      0.87       225

    accuracy                           0.65       450
   macro avg       0.47      0.45      0.45       450
weighted avg       0.62      0.65      0.63       450

Confusion matrix:
[[  1  13   7   5]
 [  2  30  43   7]
 [  2  29  58  28]
 [  0   8  15 202]]
```

- Nonevent is reliable: 87-90% recall, 85% precision
- Ia remains the hardest class: Only 4-19% recall
- Ib and II are frequently confused: 34-46 Ib predicted as II across models
- Ridge has best Ia performance: 5 correct (19% recall) vs Random Forest 1 correct (4%)
- Lasso predicts II most aggressively: Highest II recall (59%) but lowest Ib recall (29%)
- Random Forest has lowest variance (std 0.008) but worst overall class balance

#### 5.2.2 Submission

The accuracy was poor, but let's see what score we get with Logistic Regression with Ridge that showd the best per-class balance.

```python
df_x_train = df_train[features]
df_y_train = df_train["class4"]
df_x_test = df_test[features]

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)),
    ]
)

model.fit(df_x_train, df_y_train)

probs = model.predict_proba(df_x_test)
class4_pred = model.predict(df_x_test)

p_event = 1 - probs[:, list(model.classes_).index("nonevent")]

submission = pd.DataFrame({"id": df_test["id"], "class4": class4_pred, "p": p_event})
submission.to_csv("./data/submission2.csv", index=False)
```

#### 5.2.3 Result

Kaggle Score: 0.75238

Slightly worse than Submission #1.

### 5.3 Submission 3

#### 5.3.1 Approach

Frustrated by the multi-class results, I tried a really quick two-stage approach:
- Stage 1: Binary classifier using LR Lasso
- Stage 2: Event-only classifier (Ia, Ib, II) using LR Ridge trained only on event samples

#### 5.3.2 Result

Kaggle Score: 0.76921

YES!

### 5.4 Submission 4

#### 5.4.1 Approach

Hyperparameter tuning on C parameter for both stages:
- Binary tested 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, best was C=0.5 (0.893 vs 0.891)
- Events tested same range, best was C=0.1 (0.591 vs 0.551)

#### 5.4.2 Result

Kaggle Score: 0.76162

### 5.5 Submission 5

#### 5.5.1 Approach

Focused on optimizing perplexity of the score formula.

#### 5.5.2 Result

Kaggle Score: worse than submission 0.76714

## 6. Discussion and Next Steps

## References
