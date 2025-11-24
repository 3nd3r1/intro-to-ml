import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB


ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

df_test = pd.read_csv("toy_test.csv").dropna()


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def optimal_bayes_predict(x1, x2):
    z = 0.1 - 2 * x1 + x2 + 0.2 * x1 * x2
    prob_class_1 = sigmoid(z)
    predictions = (prob_class_1 > 0.5).astype(int)
    return predictions, prob_class_1


def calculate_perplexity(y_true, y_proba):
    ll = log_loss(y_true, y_proba)
    return np.exp(ll)


results_accuracy = []
results_perplexity = []

for n in ns:
    df_train = pd.read_csv(f"toy_train_{n}.csv").dropna()

    nb_model = GaussianNB()
    nb_model.fit(df_train[["x1", "x2"]], df_train["y"])
    nb_pred = nb_model.predict(df_test[["x1", "x2"]])
    nb_proba = nb_model.predict_proba(df_test[["x1", "x2"]])
    nb_accuracy = accuracy_score(df_test["y"], nb_pred)
    nb_perplexity = calculate_perplexity(df_test["y"].values, nb_proba)

    lr_model = LogisticRegression()
    lr_model.fit(df_train[["x1", "x2"]], df_train["y"])
    lr_pred = lr_model.predict(df_test[["x1", "x2"]])
    lr_proba = lr_model.predict_proba(df_test[["x1", "x2"]])
    lr_accuracy = accuracy_score(df_test["y"], lr_pred)
    lr_perplexity = calculate_perplexity(df_test["y"].values, lr_proba)

    lri_model = LogisticRegression()
    lri_model.fit(
        np.column_stack(
            [df_train["x1"], df_train["x2"], df_train["x1"] * df_train["x2"]]
        ),
        df_train["y"],
    )
    lri_pred = lri_model.predict(
        np.column_stack([df_test["x1"], df_test["x2"], df_test["x1"] * df_test["x2"]])
    )
    lri_proba = lri_model.predict_proba(
        np.column_stack([df_test["x1"], df_test["x2"], df_test["x1"] * df_test["x2"]])
    )
    lri_accuracy = accuracy_score(df_test["y"], lri_pred)
    lri_perplexity = calculate_perplexity(df_test["y"].values, lri_proba)

    optimal_bayes_pred, optimal_bayes_proba = optimal_bayes_predict(
        df_test["x1"], df_test["x2"]
    )
    optimal_bayes_accuracy = accuracy_score(df_test["y"], optimal_bayes_pred)
    optimal_bayes_proba_matrix = np.column_stack(
        [1 - optimal_bayes_proba, optimal_bayes_proba]
    )
    optimal_bayes_perplexity = calculate_perplexity(
        df_test["y"].values, optimal_bayes_proba_matrix
    )

    dummy_model = LogisticRegression(max_iter=1000, random_state=67)
    dummy_model.fit(np.zeros((len(df_train), 1)), df_train["y"])
    dummy_pred = dummy_model.predict(np.zeros((len(df_test), 1)))
    dummy_proba = dummy_model.predict_proba(np.zeros((len(df_test), 1)))
    dummy_accuracy = accuracy_score(df_test["y"], dummy_pred)
    dummy_perplexity = calculate_perplexity(df_test["y"].values, dummy_proba)

    results_accuracy.append(
        {
            "n": n,
            "NB": nb_accuracy,
            "LR": lr_accuracy,
            "LRi": lri_accuracy,
            "OptimalBayes": optimal_bayes_accuracy,
            "Dummy": dummy_accuracy,
        }
    )
    results_perplexity.append(
        {
            "n": n,
            "NB": nb_perplexity,
            "LR": lr_perplexity,
            "LRi": lri_perplexity,
            "OptimalBayes": optimal_bayes_perplexity,
            "Dummy": dummy_perplexity,
        }
    )

df_accuracy = pd.DataFrame(results_accuracy)
df_perplexity = pd.DataFrame(results_perplexity)

print(df_accuracy.to_latex(index=False, float_format="%.4f", escape=True))
print(df_perplexity.to_latex(index=False, float_format="%.4f", escape=True))
