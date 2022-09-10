import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

path = "Results\\Final\\"

print("Starting")

df = pd.read_csv(path + "UnivarARIMAcomplete.csv")
plt.plot(df["k"], df["test MSE Day"], c="#00FF00")  # green
plt.plot(df["k"], df["test MSE Night"], c="#008000")  # dark green
df = pd.read_csv(path + "MultivarARIMAcomplete.csv")
df = df.drop([i for i in range(16, 24)], axis=0)
plt.plot(df["k"], df["test MSE Day"], c="#CCCC00")  # yellow
plt.plot(df["k"], df["test MSE Night"], c="#999900")  # dark yellow
plt.title("Comparison of univariate and multivariate ARIMAs")
plt.ylabel("MSE")
plt.xlabel("k")
plt.legend(["Day Univariate", "Night Univariate", "Day Multivariate", "Night Multivariate"])
plt.savefig(path + "comparefig.png")

