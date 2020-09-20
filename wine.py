import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)
# 不均衡データであるため、yのラベルを付け直す
# 具体的にはワインの品質を3段階に分類する
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("正解率 = ", accuracy_score(y_test, y_pred))