import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


df_train = pd.read_csv("penguins_train.csv").dropna()
df_test = pd.read_csv("penguins_test.csv").dropna()

df_train["is_adelie"] = (df_train["species"] == "Adelie").astype(int)
df_test["is_adelie"] = (df_test["species"] == "Adelie").astype(int)

log_reg = smf.logit(
    "is_adelie ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g",
    data=df_train,
).fit()


coefs = log_reg.params

probs_train = log_reg.predict(df_train)
probs_test = log_reg.predict(df_test)

lrs_train = log_reg.predict(df_train, which="linear")

preds_train = (probs_train >= 0.5).astype(int)
preds_test = (probs_test >= 0.5).astype(int)

acc_train = (preds_train == df_train["is_adelie"]).mean()
acc_test = (preds_test == df_test["is_adelie"]).mean()

print(f"Coefficients:\n{coefs}\n")
print(f"Training Accuracy: {acc_train:.4f}")
print(f"Testing Accuracy: {acc_test:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(
    lrs_train[df_train["species"] == "Adelie"],
    probs_train[df_train["species"] == "Adelie"],
    color="orange",
    label="Adelie",
)
plt.scatter(
    lrs_train[df_train["species"] != "Adelie"],
    probs_train[df_train["species"] != "Adelie"],
    color="blue",
    label="Not Adelie",
)

plt.xlabel("Linear Response", fontsize=12)
plt.ylabel("Probability is Adelie", fontsize=12)
plt.title("Probability vs Linear Response", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p8-a.png")
plt.show()
