import pandas as pd
import statsmodels.api as sm


results = []

for i in range(1, 5):
    data = pd.read_csv(f"d{i}.csv")
    model = sm.OLS(data["y"], sm.add_constant(data["x"])).fit()

    intercept = model.params.iloc[0]
    intercept_se = model.bse.iloc[0]
    intercept_pval = model.pvalues.iloc[0]

    slope = model.params.iloc[1]
    slope_se = model.bse.iloc[1]
    slope_pval = model.pvalues.iloc[1]

    r_squared = model.rsquared

    results.append(
        [
            f"d{i}.csv",
            intercept,
            intercept_se,
            intercept_pval,
            slope,
            slope_se,
            slope_pval,
            r_squared,
        ]
    )

print(
    pd.DataFrame(
        results,
        columns=[
            "data",
            "intercept",
            "intercept_se",
            "intercept_pval",
            "slope",
            "slope_se",
            "slope_pval",
            "r_squared",
        ],
    ).to_latex(index=False, float_format="%.4f")
)
