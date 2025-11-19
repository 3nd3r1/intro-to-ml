import pandas as pd


df_train = pd.read_csv("penguins_train.csv").dropna()
df_test = pd.read_csv("penguins_test.csv").dropna()

stats = df_train.groupby("species").agg(["mean", "std"]).T
stats.columns.name = None
stats = stats.reset_index()
stats.columns = ["Feature", "Statistic", "Adelie", "notAdelie"]

class_counts = df_train["species"].value_counts()
class_probs = (class_counts + 1) / (len(df_train) + 2)

print(stats.to_latex(escape=True, index=False, float_format="%.4f"))
print(class_probs.to_latex(escape=True, index=True, float_format="%.4f"))
