import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./x.csv")
variables = df.var().sort_values(ascending=False).take([0,1]).keys().to_list()

df.plot(x=variables[0], y=variables[1], kind="scatter")
plt.show()
