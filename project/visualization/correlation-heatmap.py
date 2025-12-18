import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

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

corr_matrix = df_train[example_features].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0)

plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("figures/correlation-heatmap.png", dpi=300)
plt.show()
