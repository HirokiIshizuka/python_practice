import matplotlib.pyplot as plt
import pandas as pd

wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

count_data = wine.groupby('quality')["quality"].count()
print(count_data)

count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()