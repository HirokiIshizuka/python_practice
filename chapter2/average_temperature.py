import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("kion15y.csv", encoding="utf-8")

g = df.groupby(['月'])['気温']
gg = g.sum() / g.count()

print(gg)
gg.plot()
# plt.savefig("average-temperature-each-month.png")
plt.show()