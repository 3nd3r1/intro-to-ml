import pandas as pd


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

# General
print(f"taining samples: {len(df_train)}")
print(f"test samples: {len(df_test)}")
print(f"number of features: {df_train.shape[1]}")
print(f"first rows: \n{df_train.head()}")

# Class distribution
print("class4 distribution:")
print(df_train["class4"].value_counts().to_latex())
print("class4 proportion:")
print(df_train["class4"].value_counts(normalize=True).round(3).to_latex())

# Feature stats

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
print(df_train[example_features].describe().T.to_latex(escape=True, index=True, float_format="%.3f"))

# Feature correlation

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

# Feature distribution by class

df_train["class2"] = df_train["class4"].apply(lambda x: "nonevent" if x == "nonevent" else "event")

class2_means = df_train.groupby("class2")[example_features].mean()
class2_mean_diff = abs(class2_means.loc["event"] - class2_means.loc["nonevent"])
class2_mean_diff_normalized = class2_mean_diff / df_train[example_features].std()
class2_top_features = class2_mean_diff_normalized.sort_values(ascending=False)

print("top features by normalized mean different between classes (class2):")
for feature, diff in class2_top_features.items():
    print(f"{feature}: {diff:.3f}")
