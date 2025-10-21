import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv("./x.csv")
variances = data.select_dtypes(include=[np.number]).var()
if type(variances) is not pd.Series or len(variances) < 2:
    raise ValueError("Invalid data")

two_vars = variances.sort_values(ascending=False).take([0, 1]).keys().to_list()

data.plot(x=two_vars[0], y=two_vars[1], kind="scatter")
plt.show()
